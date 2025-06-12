function potentialEnergyPatch(X_patch, Y_noisy_patch, lambda_L, beta, patchSize) {
        let E_likelihood = 0;
        for (let i = 0; i < X_patch.length; i++) {
            E_likelihood += (X_patch[i] - Y_noisy_patch[i]) ** 2;
        }
        E_likelihood *= lambda_L;

        let E_prior = 0;
        for (let y = 0; y < patchSize; y++) {
            for (let x = 0; x < patchSize; x++) {
                const idx = y * patchSize + x;
                if (x < patchSize - 1) { // Right neighbor
                    E_prior += (X_patch[idx] - X_patch[idx + 1]) ** 2;
                }
                if (y < patchSize - 1) { // Down neighbor
                    E_prior += (X_patch[idx] - X_patch[idx + patchSize]) ** 2;
                }
            }
        }
        E_prior *= beta;
        return E_likelihood + E_prior;
    }

    function gradientPotentialEnergyPatch(X_patch, Y_noisy_patch, lambda_L, beta, patchSize) {
        const grad = new Float32Array(X_patch.length);
        for (let i = 0; i < X_patch.length; i++) {
            grad[i] = 2 * lambda_L * (X_patch[i] - Y_noisy_patch[i]);
            let prior_grad_term = 0;
            const x_coord = i % patchSize;
            const y_coord = Math.floor(i / patchSize);

            if (x_coord > 0) prior_grad_term += 2 * beta * (X_patch[i] - X_patch[i - 1]);
            if (x_coord < patchSize - 1) prior_grad_term += 2 * beta * (X_patch[i] - X_patch[i + 1]);
            if (y_coord > 0) prior_grad_term += 2 * beta * (X_patch[i] - X_patch[i - patchSize]);
            if (y_coord < patchSize - 1) prior_grad_term += 2 * beta * (X_patch[i] - X_patch[i + patchSize]);
            
            grad[i] += prior_grad_term;
        }
        return grad;
    }

    function kineticEnergy(p_patch) {
        let K = 0;
        for (let i = 0; i < p_patch.length; i++) K += p_patch[i] ** 2;
        return K / 2.0;
    }

    self.onmessage = function(event) {
        const {
            patchTask, // { patchId, startX, startY, noisyPatchData }
            params,    // { patchSize, lambda_L, beta, epsilon, L_steps, N_iter_per_patch, N_burnin_per_patch, updateFreq }
        } = event.data;

        const { patchId, startX, startY, noisyPatchData } = patchTask;
        const { patchSize, lambda_L, beta, epsilon, L_steps, N_iter_per_patch, N_burnin_per_patch, updateFreq } = params;

        // --- HMC logic for a single patch ---
        let X_current_patch = noisyPatchData; // Directly use the passed Float32Array
        let p_current_patch = new Float32Array(patchSize * patchSize);
        const samples_patch = [];
        let accepted_count_patch = 0;

        for (let iter = 0; iter < N_iter_per_patch; iter++) {
            for (let i = 0; i < p_current_patch.length; i++) {
                let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random();
                p_current_patch[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
            }

            let X_prop_patch = Float32Array.from(X_current_patch);
            let p_prop_patch = Float32Array.from(p_current_patch);

            const U_current = potentialEnergyPatch(X_current_patch, noisyPatchData, lambda_L, beta, patchSize);
            const K_current = kineticEnergy(p_current_patch);
            const H_current = U_current + K_current;

            let grad_X_patch = gradientPotentialEnergyPatch(X_prop_patch, noisyPatchData, lambda_L, beta, patchSize);
            for (let i = 0; i < p_prop_patch.length; i++) p_prop_patch[i] -= (epsilon / 2) * grad_X_patch[i];

            for (let l = 0; l < L_steps; l++) {
                for (let i = 0; i < X_prop_patch.length; i++) X_prop_patch[i] += epsilon * p_prop_patch[i];
                // Only compute gradient if not the last leapfrog step for X position update
                if (l < L_steps - 1) { 
                    grad_X_patch = gradientPotentialEnergyPatch(X_prop_patch, noisyPatchData, lambda_L, beta, patchSize);
                    for (let i = 0; i < p_prop_patch.length; i++) p_prop_patch[i] -= epsilon * grad_X_patch[i];
                }
            }
            // Final half-step for momentum
            grad_X_patch = gradientPotentialEnergyPatch(X_prop_patch, noisyPatchData, lambda_L, beta, patchSize);
            for (let i = 0; i < p_prop_patch.length; i++) p_prop_patch[i] -= (epsilon / 2) * grad_X_patch[i];
            
            // It's generally better to clamp X_prop_patch before calculating U_prop, 
            // as U is defined over valid pixel ranges.
            for(let i=0; i < X_prop_patch.length; i++) {
                X_prop_patch[i] = Math.max(0, Math.min(255, X_prop_patch[i]));
            }

            const U_prop = potentialEnergyPatch(X_prop_patch, noisyPatchData, lambda_L, beta, patchSize);
            const K_prop = kineticEnergy(p_prop_patch);
            const H_prop = U_prop + K_prop;
            const acceptance_prob = Math.min(1, Math.exp(H_current - H_prop));

            if (Math.random() < acceptance_prob) {
                X_current_patch.set(X_prop_patch); // Efficiently copy
                accepted_count_patch++;
            }

            if (iter >= N_burnin_per_patch) {
                samples_patch.push(Float32Array.from(X_current_patch));
            }

            if ((iter + 1) % updateFreq === 0 || iter === N_iter_per_patch - 1) {
                self.postMessage({
                    type: 'log',
                    patchId: patchId,
                    message: `  Worker (Patch ${patchId}): iter ${iter+1}/${N_iter_per_patch}, Acc.Rate: ${(accepted_count_patch/(iter+1)*100).toFixed(1)}%`
                });
            }
        }

        let X_final_avg_patch = new Float32Array(patchSize * patchSize);
        if (samples_patch.length > 0) {
            for (const sample of samples_patch) {
                for (let i = 0; i < X_final_avg_patch.length; i++) X_final_avg_patch[i] += sample[i];
            }
            for (let i = 0; i < X_final_avg_patch.length; i++) X_final_avg_patch[i] /= samples_patch.length;
        } else if (N_iter_per_patch > 0) {
            X_final_avg_patch.set(X_current_patch); // Use last state
            self.postMessage({ type: 'log', patchId: patchId, message: `  Worker (Patch ${patchId}) warning: No samples collected. Using last state.`});
        } else { // N_iter is 0
            X_final_avg_patch.set(noisyPatchData);
        }

        self.postMessage({
            type: 'result',
            patchId: patchId,
            startX: startX,
            startY: startY,
            denoisedPatchData: X_final_avg_patch // This is a Float32Array
        }, [X_final_avg_patch.buffer]); // Transfer the buffer
    };
