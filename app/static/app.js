/**
 * FORA AI // GUARDBASE SECTECH CONTROLLER
 * Real-time Agent Telemetry, Flow Pipeline Animation, Contradiction Detection,
 * Source Tier Ranking, Claim Grounding, and Grounded Follow-up Interrogation
 */

document.addEventListener("DOMContentLoaded", () => {
    // Core Form & Telemetry Elements
    const researchForm = document.getElementById("researchForm");
    const topicInput = document.getElementById("topicInput");
    const submitBtn = document.getElementById("submitBtn");
    const telemetrySection = document.getElementById("telemetrySection");
    const activeTaskIdEl = document.getElementById("activeTaskId");
    const timerBadge = document.getElementById("timerBadge");
    const fleetStateBadge = document.getElementById("fleetStateBadge");
    const traceLogStream = document.getElementById("traceLogStream");

    // Flow Pipeline SVG Nodes
    const flowPlanner = document.getElementById("flow-agent-planner");
    const flowResearcher = document.getElementById("flow-agent-researcher");
    const flowCritic = document.getElementById("flow-agent-critic");
    const flowSynthesizer = document.getElementById("flow-agent-synthesizer");

    // Execution Parameters & Controls
    const maxSourcesSelect = document.getElementById("maxSourcesSelect");
    const timeoutSelect = document.getElementById("timeoutSelect");
    const cancelTaskBtn = document.getElementById("cancelTaskBtn");

    // Dossier Section
    const reportSection = document.getElementById("reportSection");
    const dossierTopicTitle = document.getElementById("dossierTopicTitle");
    const reportMarkdownContainer = document.getElementById("reportMarkdownContainer");
    const sourcesList = document.getElementById("sourcesList");
    const sourcesCountBadge = document.getElementById("sourcesCountBadge");
    const reportIterationsBadge = document.getElementById("reportIterationsBadge");
    const queryIntentBadge = document.getElementById("queryIntentBadge");
    const uncertaintyBadge = document.getElementById("uncertaintyBadge");
    const contradictionBox = document.getElementById("contradictionAlertBox");
    const contradictionsList = document.getElementById("contradictionsList");
    const claimsList = document.getElementById("claimsList");
    const claimsCountBadge = document.getElementById("claimsCountBadge");
    const copyMarkdownBtn = document.getElementById("copyMarkdownBtn");
    const downloadMdBtn = document.getElementById("downloadMdBtn");
    const printDossierBtn = document.getElementById("printDossierBtn");

    // Session History Archive Elements
    const openHistoryBtn = document.getElementById("openHistoryBtn");
    const closeHistoryBtn = document.getElementById("closeHistoryBtn");
    const historyModal = document.getElementById("historyModal");
    const historyItemsContainer = document.getElementById("historyItemsContainer");
    const historyCountBadge = document.getElementById("historyCountBadge");
    const historyDrawerCount = document.getElementById("historyDrawerCount");

    // Grounded Q&A Elements
    const qaForm = document.getElementById("qaForm");
    const qaInput = document.getElementById("qaInput");
    const qaSubmitBtn = document.getElementById("qaSubmitBtn");
    const qaMessagesContainer = document.getElementById("qaMessagesContainer");

    // State Variables
    let currentTaskId = null;
    let pollInterval = null;
    let timerInterval = null;
    let secondsElapsed = 0;
    let latestReport = null;
    let lastRenderedLogsCount = 0;
    let selectedDepth = "quick";

    // Depth Switcher
    const depthButtons = document.querySelectorAll(".depth-toggle-btn, .depth-option");
    depthButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            depthButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            selectedDepth = btn.getAttribute("data-depth") || "quick";
        });
    });

    // Target Directive Chips
    const targetChips = document.querySelectorAll(".target-pill, .target-chip");
    targetChips.forEach(chip => {
        chip.addEventListener("click", () => {
            const prompt = chip.getAttribute("data-prompt");
            if (prompt && topicInput) {
                topicInput.value = prompt;
                topicInput.focus();
                researchForm.dispatchEvent(new Event("submit"));
            }
        });
    });

    // Dispatch Research Directive
    if (researchForm) {
        researchForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const topic = topicInput ? topicInput.value.trim() : "";
            if (!topic) return;

            // Reset UI state
            if (submitBtn) {
                submitBtn.disabled = true;
                const caption = submitBtn.querySelector(".btn-caption");
                if (caption) caption.textContent = "Researching...";
            }
            if (reportSection) reportSection.style.display = "none";
            if (telemetrySection) {
                telemetrySection.style.display = "block";
                telemetrySection.scrollIntoView({ behavior: "smooth" });
            }

            resetFlowAgents();
            if (traceLogStream) traceLogStream.innerHTML = "";
            lastRenderedLogsCount = 0;
            startTimer();

            if (fleetStateBadge) fleetStateBadge.textContent = "SEARCHING";

            const maxSources = maxSourcesSelect ? parseInt(maxSourcesSelect.value, 10) : 10;
            const timeoutSecs = timeoutSelect ? parseInt(timeoutSelect.value, 10) : 120;

            try {
                const res = await fetch("/api/v1/research", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        topic, 
                        depth: selectedDepth,
                        max_sources: maxSources,
                        time_limit_seconds: timeoutSecs
                    }),
                });

                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    throw new Error(errData.detail || `Server error: ${res.status}`);
                }

                const data = await res.json();
                currentTaskId = data.task_id;
                if (activeTaskIdEl) {
                    activeTaskIdEl.textContent = currentTaskId.substring(0, 14) + "...";
                }
                showToast("Research started. Gathering live sources...");

                startPolling(currentTaskId);
                fetchSessionHistory();
            } catch (err) {
                console.error("Failed to start research:", err);
                showToast(`Search failed: ${err.message}`, true);
                if (submitBtn) {
                    submitBtn.disabled = false;
                    const caption = submitBtn.querySelector(".btn-caption");
                    if (caption) caption.textContent = "Start Research";
                }
                if (fleetStateBadge) fleetStateBadge.textContent = "ERROR";
                stopTimer();
            }
        });
    }

    // Cancel Running Task Handler
    if (cancelTaskBtn) {
        cancelTaskBtn.addEventListener("click", async () => {
            if (!currentTaskId) return;
            try {
                cancelTaskBtn.disabled = true;
                const span = cancelTaskBtn.querySelector("span:last-child");
                if (span) span.textContent = "ABORTING...";

                const res = await fetch(`/api/v1/research/${currentTaskId}/cancel`, { method: "POST" });
                const cData = await res.json().catch(() => ({}));
                showToast(cData.message || "Termination signal sent to runtime.");
            } catch (err) {
                console.error("Cancel failed:", err);
                showToast("Failed to abort task.", true);
            } finally {
                cancelTaskBtn.disabled = false;
                const span = cancelTaskBtn.querySelector("span:last-child");
                if (span) span.textContent = "CANCEL TASK ✕";
            }
        });
    }

    // Polling Telemetry Feed
    function startPolling(taskId) {
        if (pollInterval) clearInterval(pollInterval);
        if (cancelTaskBtn) cancelTaskBtn.style.display = "inline-flex";

        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/v1/research/${taskId}`);
                if (!res.ok) return;

                const statusData = await res.json();
                updateFlowNodes(statusData);

                if (statusData.status === "completed") {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    stopTimer();
                    if (cancelTaskBtn) cancelTaskBtn.style.display = "none";
                    if (fleetStateBadge) {
                        fleetStateBadge.textContent = "COMPLETE";
                        fleetStateBadge.style.color = "var(--brand-primary)";
                    }
                    if (submitBtn) {
                        submitBtn.disabled = false;
                        const caption = submitBtn.querySelector(".btn-caption");
                        if (caption) caption.textContent = "Start Research";
                    }
                    fetchAndRenderDossier(taskId);
                    fetchSessionHistory();
                } else if (statusData.status === "cancelled") {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    stopTimer();
                    if (cancelTaskBtn) cancelTaskBtn.style.display = "none";
                    if (fleetStateBadge) {
                        fleetStateBadge.textContent = "CANCELLED";
                        fleetStateBadge.style.color = "var(--accent-red)";
                    }
                    showToast("Research cancelled.", true);
                    if (submitBtn) {
                        submitBtn.disabled = false;
                        const caption = submitBtn.querySelector(".btn-caption");
                        if (caption) caption.textContent = "Start Research";
                    }
                    fetchSessionHistory();
                } else if (statusData.status === "failed") {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    stopTimer();
                    if (cancelTaskBtn) cancelTaskBtn.style.display = "none";
                    if (fleetStateBadge) {
                        fleetStateBadge.textContent = "FAILED";
                        fleetStateBadge.style.color = "var(--accent-red)";
                    }
                    showToast(`Research failed: ${statusData.error || "Unknown error"}`, true);
                    if (submitBtn) {
                        submitBtn.disabled = false;
                        const caption = submitBtn.querySelector(".btn-caption");
                        if (caption) caption.textContent = "Start Research";
                    }
                    fetchSessionHistory();
                }
            } catch (err) {
                console.warn("Telemetry polling warning:", err);
            }
        }, 1100);
    }

    // Update Flow Pipeline Nodes
    function updateFlowNodes(data) {
        const step = data.current_step;
        const logs = data.logs || [];

        resetFlowAgents();

        if (step === "planner") {
            if (flowPlanner) flowPlanner.classList.add("active");
        } else if (step === "researcher") {
            if (flowPlanner) flowPlanner.classList.add("completed");
            if (flowResearcher) flowResearcher.classList.add("active");
        } else if (step === "critic") {
            if (flowPlanner) flowPlanner.classList.add("completed");
            if (flowResearcher) flowResearcher.classList.add("completed");
            if (flowCritic) flowCritic.classList.add("active");
        } else if (step === "synthesizer" || step === "verifier") {
            if (flowPlanner) flowPlanner.classList.add("completed");
            if (flowResearcher) flowResearcher.classList.add("completed");
            if (flowCritic) flowCritic.classList.add("completed");
            if (flowSynthesizer) flowSynthesizer.classList.add("active");
        } else if (step === "completed") {
            if (flowPlanner) flowPlanner.classList.add("completed");
            if (flowResearcher) flowResearcher.classList.add("completed");
            if (flowCritic) flowCritic.classList.add("completed");
            if (flowSynthesizer) flowSynthesizer.classList.add("completed");
        }

        // Stream New Logs
        if (logs.length > lastRenderedLogsCount && traceLogStream) {
            for (let i = lastRenderedLogsCount; i < logs.length; i++) {
                appendFeedLog(logs[i]);
            }
            lastRenderedLogsCount = logs.length;
        }
    }

    function resetFlowAgents() {
        [flowPlanner, flowResearcher, flowCritic, flowSynthesizer].forEach(node => {
            if (node) {
                node.classList.remove("active");
                node.classList.remove("completed");
            }
        });
    }

    function appendFeedLog(log) {
        const div = document.createElement("div");
        div.className = "log-entry";
        const nodeName = (log.node || "AGENT").toLowerCase();
        div.innerHTML = `
            <span class="log-time">${log.time || ""}</span>
            <span class="log-badge ${nodeName}">${(log.node || "AGENT").toUpperCase()}</span>
            <span class="log-msg">${escapeHtml(log.action || "")}</span>
        `;
        traceLogStream.appendChild(div);
        traceLogStream.scrollTop = traceLogStream.scrollHeight;
    }

    // Render Final Intelligence Dossier
    async function fetchAndRenderDossier(taskId) {
        try {
            const res = await fetch(`/api/v1/research/${taskId}/report`);
            if (!res.ok) throw new Error("Could not retrieve completed dossier.");

            const data = await res.json();
            latestReport = data;
            const citationsList = data.citations || data.sources || [];
            const markdownBody = data.markdown_content || data.report || "";

            if (dossierTopicTitle) {
                dossierTopicTitle.textContent = data.topic || "Executive Intelligence Briefing";
            }
            if (sourcesCountBadge) {
                sourcesCountBadge.textContent = `${citationsList.length} RANKED SOURCES`;
            }
            if (reportIterationsBadge) {
                reportIterationsBadge.textContent = `${data.iterations_used || data.iteration_count || 1} AUDIT CYCLES`;
            }

            // Set Intent & Certainty Badges
            const intent = (data.query_profile && data.query_profile.intent) ? data.query_profile.intent : "EXPLANATORY";
            if (queryIntentBadge) {
                queryIntentBadge.textContent = intent.replace(/_/g, " ");
            }

            const cert = data.overall_uncertainty || "WELL_SUPPORTED";
            if (uncertaintyBadge) {
                uncertaintyBadge.textContent = cert.replace(/_/g, " ");
                uncertaintyBadge.style.color = cert === "WELL_SUPPORTED" ? "var(--brand-primary)" : cert === "CONFLICTING_EVIDENCE" ? "var(--accent-yellow)" : "#f87171";
            }

            // Render Contradictions Alert Box
            if (data.contradictions && data.contradictions.length > 0) {
                if (contradictionBox) contradictionBox.style.display = "block";
                if (contradictionsList) {
                    contradictionsList.innerHTML = data.contradictions.map(c => `
                        <div class="contradiction-item">
                            <strong>${escapeHtml(c.topic || "Disagreement")}:</strong> 
                            Source A claims <em>"${escapeHtml(c.source_a_claim || "")}"</em> vs 
                            Source B claims <em>"${escapeHtml(c.source_b_claim || "")}"</em>.<br>
                            <span style="color: #94a3b8; font-size: 12px;"><strong>Analysis:</strong> ${escapeHtml(c.discrepancy_analysis || "")}</span>
                        </div>
                    `).join("");
                }
            } else {
                if (contradictionBox) contradictionBox.style.display = "none";
            }

            // Render Markdown body
            if (reportMarkdownContainer) {
                reportMarkdownContainer.innerHTML = renderMarkdown(markdownBody || "No content generated.");
            }

            // Render Claim Grounding Drawer
            if (claimsCountBadge) {
                claimsCountBadge.textContent = `${data.claims ? data.claims.length : 0} AUDITED CLAIMS`;
            }
            if (claimsList) {
                claimsList.innerHTML = "";
                if (data.claims && data.claims.length > 0) {
                    data.claims.forEach(clm => {
                        const pillClass = clm.support_level === "WELL_SUPPORTED" ? "well" : clm.support_level === "PARTIALLY_SUPPORTED" ? "partial" : clm.support_level === "CONFLICTING_EVIDENCE" ? "conflict" : "insufficient";
                        const div = document.createElement("div");
                        div.className = "claim-row";
                        const quoteHtml = clm.supporting_quote ? `<div class="claim-quote">"${escapeHtml(clm.supporting_quote)}"</div>` : "";
                        const srcLinks = (clm.source_urls || []).map((u, i) => `<a href="${escapeHtml(u)}" target="_blank" rel="noopener noreferrer">[Source ${i+1}]</a>`).join(" ");
                        div.innerHTML = `
                            <div class="claim-meta-row">
                                <span class="claim-id">[${escapeHtml(clm.claim_id || "CLM")}]</span>
                                <span class="support-pill ${pillClass}">${escapeHtml((clm.support_level || "").replace(/_/g, " "))}</span>
                            </div>
                            <div class="claim-text">${escapeHtml(clm.claim_text || "")}</div>
                            ${quoteHtml}
                            <div class="claim-sources-list">Backing: ${srcLinks || "Directly inferred from validated evidence"}</div>
                        `;
                        claimsList.appendChild(div);
                    });
                } else {
                    claimsList.innerHTML = `<div style="color: var(--text-muted); font-size: 13px; font-family: var(--font-mono);">Evidence grounded across synthesized canonical sources.</div>`;
                }
            }

            // Render Citation Cards with Source Quality Tier
            if (sourcesList) {
                sourcesList.innerHTML = "";
                if (citationsList.length > 0) {
                    citationsList.forEach((src, idx) => {
                        const card = document.createElement("a");
                        card.className = "source-card-gb";
                        card.href = src.url;
                        card.target = "_blank";
                        card.rel = "noopener noreferrer";

                        let domain = src.domain || "web";
                        if (!src.domain) {
                            try {
                                domain = new URL(src.url).hostname;
                            } catch (e) {
                                domain = src.url.substring(0, 30);
                            }
                        }

                        const tier = (src.tier || "TIER_2").replace("TIER_", "");
                        const tierClass = tier.startsWith("1") ? "tier-1" : tier.startsWith("2") ? "tier-2" : "tier-3";
                        const tierName = tier.startsWith("1") ? "PRIMARY" : tier.startsWith("2") ? "SECONDARY" : "TERTIARY";

                        card.innerHTML = `
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="source-index">[SOURCE ${String(idx + 1).padStart(2, '0')}]</span>
                                <span class="tier-pill ${tierClass}">${tierName}</span>
                            </div>
                            <span class="source-title" title="${escapeHtml(src.title || src.url)}">${escapeHtml(src.title || "Reference Document")}</span>
                            <span class="source-domain">${domain}</span>
                        `;
                        sourcesList.appendChild(card);
                    });
                } else {
                    sourcesList.innerHTML = `<div style="color: var(--text-muted); font-size: 13px; font-family: var(--font-mono);">No external citation references recorded.</div>`;
                }
            }

            // Display Dossier & Reset Submit
            if (reportSection) {
                reportSection.style.display = "block";
                reportSection.scrollIntoView({ behavior: "smooth" });
            }
            if (submitBtn) {
                submitBtn.disabled = false;
                const caption = submitBtn.querySelector(".btn-caption");
                if (caption) caption.textContent = "DISPATCH AGENTS";
            }

            showToast("Intelligence dossier compiled and verified.");
        } catch (err) {
            console.error("Dossier render failure:", err);
            showToast("Failed to render report dossier.", true);
            if (submitBtn) {
                submitBtn.disabled = false;
                const caption = submitBtn.querySelector(".btn-caption");
                if (caption) caption.textContent = "DISPATCH AGENTS";
            }
        }
    }

    // Grounded Q&A Follow-up Interrogation
    if (qaForm) {
        qaForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const question = qaInput ? qaInput.value.trim() : "";
            if (!question || !currentTaskId) return;

            // Render User Bubble
            appendQaBubble("user", question);
            if (qaInput) qaInput.value = "";
            if (qaSubmitBtn) {
                qaSubmitBtn.disabled = true;
                qaSubmitBtn.querySelector("span").textContent = "Inquiring...";
            }

            try {
                const res = await fetch(`/api/v1/research/${currentTaskId}/ask`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question }),
                });

                if (!res.ok) {
                    const errData = await res.json().catch(() => ({}));
                    throw new Error(errData.detail || "Grounded query failed.");
                }

                const data = await res.json();
                appendQaBubble("agent", data.answer, data.grounded_sources);
            } catch (err) {
                console.error("Grounded Q&A Error:", err);
                appendQaBubble("agent", `Inquiry error: ${err.message}. Grounded verification layer prevented hallucinated response.`);
            } finally {
                if (qaSubmitBtn) {
                    qaSubmitBtn.disabled = false;
                    qaSubmitBtn.querySelector("span").textContent = "Inquire";
                }
            }
        });
    }

    function appendQaBubble(sender, text, citations = []) {
        if (!qaMessagesContainer) return;

        const bubble = document.createElement("div");
        bubble.className = `qa-bubble ${sender}`;

        let citationsHtml = "";
        if (citations && citations.length > 0) {
            citationsHtml = `
                <div class="qa-citations-list">
                    <strong>VERIFIED EVIDENCE CITATIONS:</strong><br>
                    ${citations.map(c => typeof c === 'string' ? `• <a href="${escapeHtml(c)}" target="_blank" style="color: var(--brand-primary); text-decoration: underline;">${escapeHtml(c)}</a>` : `• <a href="${escapeHtml(c.url)}" target="_blank" style="color: var(--brand-primary); text-decoration: underline;">${escapeHtml(c.title || c.url)}</a>`).join("<br>")}
                </div>
            `;
        }

        bubble.innerHTML = `
            <div class="qa-bubble-tag">${sender === "user" ? "INVESTIGATOR" : "FORA // ZERO-TEMP AUDITOR"}</div>
            <div>${escapeHtml(text).replace(/\n/g, "<br>")}</div>
            ${citationsHtml}
        `;

        qaMessagesContainer.appendChild(bubble);
        qaMessagesContainer.scrollTop = qaMessagesContainer.scrollHeight;
    }

    // Copy Raw Markdown
    if (copyMarkdownBtn) {
        copyMarkdownBtn.addEventListener("click", () => {
            const rawContent = latestReport ? (latestReport.markdown_content || latestReport.report || "") : "";
            if (!rawContent) return;
            navigator.clipboard.writeText(rawContent).then(() => {
                showToast("Raw markdown copied to clipboard.");
            }).catch(err => {
                showToast("Failed to copy markdown.", true);
            });
        });
    }

    // Download .MD File
    if (downloadMdBtn) {
        downloadMdBtn.addEventListener("click", () => {
            const rawContent = latestReport ? (latestReport.markdown_content || latestReport.report || "") : "";
            if (!rawContent) return;
            const blob = new Blob([rawContent], { type: "text/markdown;charset=utf-8;" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            const safeTopic = (latestReport.topic || "research-report").replace(/[^a-z0-9]/gi, '_').toLowerCase();
            a.download = `fora_${safeTopic}.md`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showToast("Report exported as Markdown.");
        });
    }

    // Print / Clean Vector PDF Export
    if (printDossierBtn) {
        printDossierBtn.addEventListener("click", () => {
            if (!latestReport) {
                showToast("No active dossier available to print.", true);
                return;
            }
            window.print();
        });
    }

    // Session Mission History Archive
    async function fetchSessionHistory() {
        try {
            const res = await fetch("/api/v1/research");
            if (!res.ok) return;
            const tasks = await res.json();

            if (historyCountBadge) {
                historyCountBadge.textContent = tasks.length;
            }
            if (historyDrawerCount) {
                historyDrawerCount.textContent = `${tasks.length} MISSIONS RECORDED IN CURRENT SESSION`;
            }

            if (!historyItemsContainer) return;
            historyItemsContainer.innerHTML = "";

            if (!tasks || tasks.length === 0) {
                historyItemsContainer.innerHTML = `
                    <div style="padding: 24px 0; text-align: center; color: var(--text-muted); font-size: 13px; font-family: var(--font-mono);">
                        NO RESEARCH MISSIONS RECORDED YET. DISPATCH A DIRECTIVE TO POPULATE ARCHIVE.
                    </div>
                `;
                return;
            }

            tasks.forEach(t => {
                const card = document.createElement("div");
                card.className = "history-mission-card";
                if (t.task_id === currentTaskId) {
                    card.classList.add("active-mission");
                }

                const statusClass = t.status === "completed" ? "completed" : t.status === "cancelled" ? "cancelled" : t.status === "failed" ? "failed" : "running";
                const dateStr = t.created_at ? new Date(t.created_at).toLocaleTimeString() : "--:--";

                card.innerHTML = `
                    <div class="mission-top">
                        <span class="mission-topic">${escapeHtml(t.topic || "Untitled Mission")}</span>
                        <span class="mission-status-pill ${statusClass}">${escapeHtml((t.status || "").toUpperCase())}</span>
                    </div>
                    <div class="mission-meta">
                        <span>DEPTH: <strong>${escapeHtml((t.depth || "quick").toUpperCase())}</strong></span>
                        <span>AUDIT CYCLES: <strong>${t.iteration_count || 1}</strong></span>
                        <span>DISPATCHED: <strong>${dateStr}</strong></span>
                    </div>
                `;

                card.addEventListener("click", () => {
                    if (t.status === "completed") {
                        currentTaskId = t.task_id;
                        if (activeTaskIdEl) {
                            activeTaskIdEl.textContent = t.task_id.substring(0, 14) + "...";
                        }
                        fetchAndRenderDossier(t.task_id);
                        closeHistoryDrawer();
                        if (reportSection) {
                            reportSection.style.display = "block";
                            reportSection.scrollIntoView({ behavior: "smooth" });
                        }
                        showToast(`Loaded mission dossier: ${t.topic.substring(0, 32)}...`);
                    } else if (t.status === "in_progress") {
                        currentTaskId = t.task_id;
                        if (activeTaskIdEl) {
                            activeTaskIdEl.textContent = t.task_id.substring(0, 14) + "...";
                        }
                        closeHistoryDrawer();
                        startPolling(t.task_id);
                        if (telemetrySection) {
                            telemetrySection.style.display = "block";
                            telemetrySection.scrollIntoView({ behavior: "smooth" });
                        }
                    } else {
                        showToast(`Mission was ${t.status}. Final dossier is unavailable.`, true);
                    }
                });

                historyItemsContainer.appendChild(card);
            });
        } catch (err) {
            console.warn("Failed to fetch session history:", err);
        }
    }

    function openHistoryDrawer() {
        if (historyModal) historyModal.classList.add("open");
        fetchSessionHistory();
    }

    function closeHistoryDrawer() {
        if (historyModal) historyModal.classList.remove("open");
    }

    if (openHistoryBtn) {
        openHistoryBtn.addEventListener("click", (e) => {
            e.preventDefault();
            openHistoryDrawer();
        });
    }

    if (closeHistoryBtn) {
        closeHistoryBtn.addEventListener("click", closeHistoryDrawer);
    }

    if (historyModal) {
        historyModal.addEventListener("click", (e) => {
            if (e.target === historyModal) {
                closeHistoryDrawer();
            }
        });
    }

    // Initialize session history badge on load
    fetchSessionHistory();

    // Timer Functions
    function startTimer() {
        stopTimer();
        secondsElapsed = 0;
        updateTimerDisplay();
        timerInterval = setInterval(() => {
            secondsElapsed++;
            updateTimerDisplay();
        }, 1000);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }

    function updateTimerDisplay() {
        if (!timerBadge) return;
        const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, "0");
        const secs = String(secondsElapsed % 60).padStart(2, "0");
        timerBadge.textContent = `${mins}:${secs}`;
    }

    // Toast Notification Utility
    function showToast(message, isError = false) {
        const existing = document.querySelector(".toast-notice");
        if (existing) existing.remove();

        const toast = document.createElement("div");
        toast.className = "toast-notice";
        if (isError) {
            toast.style.borderColor = "var(--accent-red)";
            toast.style.color = "#fca5a5";
        }
        toast.innerHTML = `
            <span>${isError ? "✕" : "✓"}</span>
            <span>${escapeHtml(message)}</span>
        `;
        document.body.appendChild(toast);

        setTimeout(() => {
            if (toast) toast.remove();
        }, 3600);
    }

    // Markdown Parser with Table Support
    function renderMarkdown(md) {
        if (!md) return "";
        let html = escapeHtml(md);

        // Code blocks
        html = html.replace(/```([a-z]*)\n([\s\S]*?)```/gi, (match, lang, code) => {
            return `<pre style="background: #090b0e; border: 1px solid var(--border-default); padding: 14px; font-family: var(--font-mono); font-size: 12px; overflow-x: auto; margin: 16px 0;"><code>${code}</code></pre>`;
        });

        // Markdown Tables (Comparison Matrices)
        html = html.replace(/\|(.+)\|[\r\n]+\|[-:| ]+\|[\r\n]+((?:\|.*\|[\r\n]*)+)/g, (match, headerLine, bodyLines) => {
            const headers = headerLine.split('|').map(h => h.trim()).filter(h => h.length > 0);
            const rows = bodyLines.trim().split('\n').map(row => {
                const cells = row.split('|').map(c => c.trim()).filter(c => c.length > 0);
                return `<tr>${cells.map(c => `<td>${c}</td>`).join('')}</tr>`;
            }).join('');
            return `<table><thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>`;
        });

        // Markdown Links
        html = html.replace(/\[(.*?)\]\((https?:\/\/[^\s\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" style="color: var(--brand-primary); text-decoration: underline;">$1</a>');

        // Headers
        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
        html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

        // Bold and Italic
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Blockquotes
        html = html.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

        // Lists
        html = html.replace(/^\s*[\-\*]\s+(.*$)/gim, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/gims, '<ul style="margin: 12px 0 16px; padding-left: 20px;">$1</ul>');

        // Paragraphs
        html = html.split('\n\n').map(p => {
            p = p.trim();
            if (!p) return '';
            if (p.startsWith('<h') || p.startsWith('<pre') || p.startsWith('<ul') || p.startsWith('<table') || p.startsWith('<blockquote')) {
                return p;
            }
            return `<p>${p}</p>`;
        }).join('');

        return html;
    }

    function escapeHtml(text) {
        if (!text) return "";
        return String(text)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
