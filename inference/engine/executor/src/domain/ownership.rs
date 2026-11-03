//! Request residency, resume states, and reclaim.

use super::*;

/// Where an opened request's state comes from.
#[derive(Clone, Copy)]
enum Opening<'a> {
    /// Fresh state at zero.
    Fresh,
    /// A resume state's, at its position.
    Resume(&'a ResumeState),
    /// Saved state exported from this domain's stores, imported into them.
    Import {
        target: &'a ExportedState,
        head: Option<&'a ExportedState>,
    },
}

impl Opening<'_> {
    /// The position the opened state is at.
    fn position(self) -> usize {
        match self {
            Self::Fresh => 0,
            Self::Resume(from) => from.position(),
            Self::Import { target, .. } => target.position,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenRequirements {
    target_banks: usize,
    head_banks: usize,
}

impl OpenRequirements {
    pub fn target_banks(self) -> usize {
        self.target_banks
    }
    pub fn head_banks(self) -> usize {
        self.head_banks
    }
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    pub fn open_requirements(&self) -> OpenRequirements {
        OpenRequirements {
            target_banks: 1,
            head_banks: usize::from(self.head_store.is_some()),
        }
    }

    pub fn can_open(&self, requirements: OpenRequirements) -> Result<(), CapacityError> {
        let target = self.target_store.available_banks();
        if target < requirements.target_banks {
            return Err(CapacityError {
                resource: ResourceKind::RecurrentBanks,
                required: requirements.target_banks as u64,
                available: target as u64,
            });
        }
        if let Some(store) = &self.head_store {
            let head = store.available_banks();
            if head < requirements.head_banks {
                return Err(CapacityError {
                    resource: ResourceKind::RecurrentBanks,
                    required: requirements.head_banks as u64,
                    available: head as u64,
                });
            }
        }
        Ok(())
    }

    /// Make a request with installed input resident: fresh state at zero, or
    /// `from`'s state at its position. Features `from` holds for an image
    /// with a span straddling that position are adopted; every image placed
    /// after it that still lacks features is returned as an encode. Opening
    /// into free banks changes no bindings; growing one concludes the queued
    /// steps first (`conclude_chain`): an open during another request's
    /// chain waits for its end (`unchained`), so as not to cut it short. Admission is new
    /// work's: refused, changing nothing, where the device's ledger cannot
    /// reserve the rows it is about to write, its `rows` (its prompt, or the
    /// history it replays), with its drafter's rows past them and a launch's
    /// rows ahead (`RequestRows::opening`, `Ledger::reserve`), so no prompt
    /// row waits on memory. Its later rows are reserved ahead at its steps'
    /// boundaries ([`Self::provision`]). The idle optional components are
    /// kept for it to draft and encode with in the same change, before its
    /// reservation claims (`Self::keep_components`), so the reservation
    /// cannot take them; refused, they idle again as they were. Opened, the
    /// request holds its reservation, backed.
    pub fn open_state(
        &mut self,
        unchained: Unchained<'_>,
        request: RequestId,
        from: Option<&ResumeState>,
        rows: u64,
    ) -> Result<Vec<Operation>, DomainError> {
        self.open(unchained, request, from.map_or(Opening::Fresh, Opening::Resume), rows)
    }

    /// Make a request with installed input resident from saved state
    /// exported from this domain's stores ([`StateCheckpoint::export`]), the
    /// SSD cold tier's: as [`Self::open_state`] opens from a resume state,
    /// the saved rows imported into the stores where it would fork one. The
    /// sequence holds the first `position` tokens of its input, and no
    /// image's features: the cold tier saves no state conditioned on one.
    pub fn restore_state(
        &mut self,
        unchained: Unchained<'_>,
        request: RequestId,
        target: &ExportedState,
        head: Option<&ExportedState>,
        rows: u64,
    ) -> Result<Vec<Operation>, DomainError> {
        self.open(unchained, request, Opening::Import { target, head }, rows)
    }

    fn open(
        &mut self,
        mut unchained: Unchained<'_>,
        request: RequestId,
        opening: Opening<'_>,
        rows: u64,
    ) -> Result<Vec<Operation>, DomainError> {
        let installed = self.input.get(&request).expect("invariant violated: an opened request has no installed input");
        assert!(
            !installed.resident && !self.target.contains_key(&request) && !self.head.contains_key(&request),
            "invariant violated: request {} opens already resident",
            request.0
        );
        let from = match opening {
            Opening::Resume(from) => Some(from),
            Opening::Fresh | Opening::Import { .. } => None,
        };
        let resumes = opening.position() as u64;
        assert!(
            resumes <= rows,
            "invariant violated: request {} resumes at row {resumes}, past the {rows} rows it opens with",
            request.0
        );
        let reserved = crate::memory::ledger::RequestRows::of(&self.execution).opening(rows);
        let changes = self.resources.ledgers().changes().clone();
        let mut reservation = changes
            .change(|keeping| {
                let kept = self.keep_components(&keeping);
                let reservation = self.resources.ledgers().device().reserve(&keeping.claims(), &self.request_slot, request, reserved);
                match reservation {
                    Ok(_) => self.activate_components(kept),
                    Err(_) => self.idle_kept(kept),
                }
                reservation
            })
            .map_err(DomainError::request_short)?;
        self.conclude_chain()?;
        let (target, head) = match opening {
            Opening::Resume(from) => self.fork_resume_state(from),
            Opening::Fresh => self.fresh_state(&mut unchained)?,
            Opening::Import { target, head } => self.import_state(&mut unchained, request, target, head)?,
        };
        let position = target.position();
        reservation.grow(position as u64);
        self.change_state(|this, _| this.state_claims.insert(request, reservation));
        self.target.insert(request, target);
        if let Some(head) = head {
            self.head.insert(request, head);
        }
        if let Err(error) = self.back_reservation(unchained.bindings(), request) {
            self.target.remove(&request);
            self.head.remove(&request);
            self.change_state(|this, _| this.drop_state_claim(request));
            return Err(error);
        }
        let installed = self.input.get_mut(&request).expect("input checked above");
        installed.resident = true;
        installed.boundary = from.and_then(|from| from.boundary.clone());
        // A resumed sequence holds the tokens of the state it resumes; an
        // imported one, those of its input before its position.
        installed.tokens = match opening {
            Opening::Resume(from) => input::TokenHistory::starting(&from.tokens),
            Opening::Import { .. } => input::TokenHistory::starting(&installed.input.tokens()[..position]),
            Opening::Fresh => Default::default(),
        };
        let spans = installed.input.layout().spans();
        let mut encodes = Vec::new();
        for (identity, slot) in &mut installed.images {
            // Spans are ordered and disjoint: only an image's next placement
            // can straddle the position.
            let Some(next) = spans
                .iter()
                .find(|span| span.identity == *identity && position < span.end)
            else {
                continue;
            };
            if next.start < position {
                slot.features = from.and_then(|from| from.features.get(identity).cloned());
            }
            if slot.features.is_none() {
                encodes.push(Operation::Encode {
                    request,
                    image: slot.image.clone(),
                });
            }
        }
        Ok(encodes)
    }

    /// A test's open of `request` about to write the load's served context:
    /// its whole context reserved ([`Self::open_state`]); refused as invariant while a
    /// chain is under way, which a test opens beside only to see deferred.
    #[cfg(any(test, feature = "fixtures"))]
    pub fn open_served(
        &mut self,
        bindings: &mut StateBindings,
        request: RequestId,
        from: Option<&ResumeState>,
    ) -> Result<Vec<Operation>, DomainError> {
        let rows = self.execution.resources().target_state().context_rows as u64;
        let unchained = self.unchained(bindings).expect("invariant violated: a test opens a request beside a chain");
        self.open_state(unchained, request, from, rows)
    }

    /// Import saved state into this domain's stores as fresh histories hold
    /// it ([`StateStore::import`]): each store first grows by the whole
    /// pages its saved rows take and the bank the import claims, as an open
    /// provisions its banks ([`Self::provision_open`]).
    fn import_state(
        &mut self,
        unchained: &mut Unchained<'_>,
        request: RequestId,
        target: &ExportedState,
        head: Option<&ExportedState>,
    ) -> Result<(SequenceState, Option<SequenceState>), DomainError> {
        if self.head_store.is_some() != head.is_some() {
            return Err(DomainError::invariant(
                "restore state head does not match the domain head arena",
            ));
        }
        let installed = self.input.get(&request).expect("invariant violated: an opened request has no installed input");
        if installed.input.tokens().len() < target.position {
            return Err(DomainError::invariant(
                "restore position exceeds the admitted input tokens",
            ));
        }
        self.conclude_chain()?;
        let requirements = self.open_requirements();
        let target_demands = self.target_store.import_demands(target);
        let head_demands = self
            .head_store
            .as_ref()
            .zip(head)
            .map(|(store, saved)| store.import_demands(saved));
        self.grant_state_growth(unchained.bindings(), false, &target_demands, &[], requirements.target_banks())?;
        if let Some(demands) = head_demands {
            self.grant_state_growth(unchained.bindings(), true, &demands, &[], requirements.head_banks())?;
        }
        let target = self.target_store.import(target)?;
        let head = self
            .head_store
            .as_ref()
            .zip(head)
            .map(|(store, saved)| store.import(saved))
            .transpose()?;
        Ok((target, head))
    }

    fn fresh_state(
        &mut self,
        unchained: &mut Unchained<'_>,
    ) -> Result<(SequenceState, Option<SequenceState>), DomainError> {
        self.provision_open(unchained)?;
        let requirements = self.open_requirements();
        self.can_open(requirements).map_err(DomainError::Capacity)?;
        let target = self.target_store.create().map_err(|error| {
            DomainError::invariant(format!(
                "target admission capacity changed after availability check: {error}"
            ))
        })?;
        let head = self
            .head_store
            .as_ref()
            .map(|store| {
                store.create().map_err(|error| {
                    DomainError::invariant(format!(
                        "head admission capacity changed after availability check: {error}"
                    ))
                })
            })
            .transpose()?;
        Ok((target, head))
    }

    fn fork_resume_state(
        &self,
        from: &ResumeState,
    ) -> (SequenceState, Option<SequenceState>) {
        let target = from.target.fork();
        assert!(
            target.belongs_to(&self.target_store) && from.head.is_some() == self.head_store.is_some(),
            "invariant violated: a resume state differs from the domain's state arenas"
        );
        let head = from.head.as_ref().map(StateCheckpoint::fork);
        assert!(
            head.as_ref().is_none_or(|state| self.head_store.as_ref().is_some_and(|store| state.belongs_to(store))),
            "invariant violated: a resume state's head belongs to another state arena"
        );
        (target, head)
    }

    /// The request's reconciled state as a resume state: both lanes and the
    /// features of spans straddling its position.
    pub fn resume_state(&self, request: RequestId) -> Result<ResumeState, String> {
        let target = self
            .target
            .get(&request)
            .ok_or_else(|| format!("request {} has in-flight work or is not open", request.0))?
            .checkpoint();
        if self.head_store.is_some() && !self.head.contains_key(&request) {
            return Err("request has unresolved head work".into());
        }
        let head = self.head.get(&request).map(SequenceState::checkpoint);
        let installed = self
            .input
            .get(&request)
            .ok_or("resident request has no installed input")?;
        let position = target.position();
        let features = installed
            .input
            .layout()
            .spans()
            .iter()
            .filter(|span| span.start < position && position < span.end)
            .map(|span| {
                let features = installed
                    .images
                    .get(&span.identity)
                    .and_then(|slot| slot.features.clone())
                    .ok_or("resident rows condition on an unencoded image")?;
                Ok((span.identity.clone(), features))
            })
            .collect::<Result<_, String>>()?;
        Ok(ResumeState {
            tokens: installed.tokens.before(position)?.into(),
            target,
            head,
            features,
            boundary: installed.boundary.clone(),
        })
    }

    /// Drop a request's input and any numerical state. A resident request's
    /// state must not be in flight; a chain holder's chain concludes first
    /// ([`Self::conclude_chain`]), its queued chunks, which still read the
    /// chain's rows, waited for.
    pub fn close(&mut self, request: RequestId) -> Result<(), String> {
        if self.chain_holder() == Some(request) {
            self.conclude_chain().map_err(|error| error.to_string())?;
        }
        let installed = self.input.get(&request).ok_or("request is not open")?;
        if installed.resident
            && (!self.target.contains_key(&request)
                || (self.head_store.is_some() && !self.head.contains_key(&request)))
        {
            return Err("request has unresolved work".into());
        }
        self.release_prefetches(&[request])?;
        self.head.remove(&request);
        self.proposed.remove(&request);
        self.publishing.remove(&request);
        self.input.remove(&request);
        self.target.remove(&request);
        self.change_state(|this, _| this.drop_state_claim(request));
        self.poisoned.remove(&request);
        Ok(())
    }

    /// Evict these requests: release their numerical state and encoded
    /// features. Their input stays installed for when they are resident
    /// again. A chain one holds concludes first ([`Self::conclude_chain`]),
    /// its state then the holder's. Refused, releasing nothing, where one
    /// holds no resolved state (a head priming not yet completed).
    pub fn release_state(&mut self, requests: &[RequestId]) -> Result<(), String> {
        if self.chain_holder().is_some_and(|holder| requests.contains(&holder)) {
            self.conclude_chain().map_err(|error| error.to_string())?;
        }
        if !requests.iter().all(|request| self.target.contains_key(request)) {
            return Err("an evicted request has unresolved or absent target state".into());
        }
        if self.head_store.is_some() && !requests.iter().all(|request| self.head.contains_key(request)) {
            return Err("an evicted request has unresolved or absent head state".into());
        }
        // Prefetched for the residency that ends; a state opened again
        // prefetches its own.
        self.release_prefetches(requests)?;
        for request in requests {
            let installed = self
                .input
                .get_mut(request)
                .ok_or("evicted request has no installed input")?;
            installed.resident = false;
            installed.boundary = None;
            for slot in installed.images.values_mut() {
                slot.features = None;
            }
            self.target.remove(request);
            self.head.remove(request);
        }
        self.change_state(|this, _| requests.iter().for_each(|request| this.drop_state_claim(*request)));
        Ok(())
    }
}
