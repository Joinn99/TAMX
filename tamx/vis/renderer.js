// JET Colormap Implementation
function getJetColor(value, alphaBase = 0.6) {
    const r = Math.min(Math.max(Math.min(4 * value - 1.5, -4 * value + 4.5), 0), 1);
    const g = Math.min(Math.max(Math.min(4 * value - 0.5, -4 * value + 3.5), 0), 1);
    const b = Math.min(Math.max(Math.min(4 * value + 0.5, -4 * value + 2.5), 0), 1);

    const alpha = value * alphaBase;
    return `rgba(${Math.floor(r * 255)}, ${Math.floor(g * 255)}, ${Math.floor(b * 255)}, ${alpha.toFixed(2)})`;
}

class TAMVisualizer {
    constructor(containerId, data) {
        this.container = document.getElementById(containerId);
        this.data = data;
        this.unpackData();
        this.selectedIdx = -1;
        this.selectedCandidateIdx = 0;
        this.hoverAnsIdx = -1;
        this.hoverTokenIdx = -1;
        this.visibleTurns = new Set();
        this.hoveredCells = {};
        this.init();
    }

    init() {
        this.container.innerHTML = `
            <div class="tam-layout">
                <div class="tam-chat-panel" id="chat-panel">
                    <div class="tam-candidate-menu" id="candidate-menu" style="display:none;"></div>
                </div>
                <div class="tam-splitter" id="tam-splitter"></div>
                <div class="tam-vision-panel" id="vision-panel"></div>
                <div id="value-tooltip" class="tam-tooltip"></div>
            </div>
        `;
        this.renderChat();
        this.renderVision();
        this.setupVisibilityTracking();
        this.setupSplitDrag();
        this.setupCandidateMenu();
    }

    renderChat() {
        const panel = document.getElementById('chat-panel');
        const candidateMenu = document.getElementById('candidate-menu');
        panel.innerHTML = '';
        if (candidateMenu) {
            panel.appendChild(candidateMenu);
        }

        let globalTokenIdx = 0;
        this.data.chat_turns.forEach((turn, turnIdx) => {
            const turnDiv = document.createElement('div');
            turnDiv.className = `tam-turn ${turn.role}`;
            turnDiv.dataset.turnIdx = turnIdx;

            const avatar = document.createElement('div');
            avatar.className = 'tam-avatar';
            if (turn.role === 'user') {
                avatar.innerText = 'U';
            } else if (turn.role === 'system') {
                avatar.innerText = 'S';
            } else {
                avatar.innerText = 'AI';
            }
            turnDiv.appendChild(avatar);

            const content = document.createElement('div');
            content.className = 'tam-message-content';

            turn.tokens.forEach((token) => {
                const span = document.createElement('span');
                span.innerText = token.text;

                if (token.type === 'vision-link') {
                    span.className = 'tam-token vision-link';
                    span.onmouseenter = () => this.highlightImage(token.vision_idx, true);
                    span.onmouseleave = () => this.highlightImage(token.vision_idx, false);
                    content.appendChild(span);
                    return;
                }

                const currentGlobalIdx = globalTokenIdx++;
                span.className = `tam-token ${token.type}`;
                span.dataset.idx = currentGlobalIdx;

                if (token.type === 'answer') {
                    const ansIdx = token.ans_idx;
                    span.onclick = (e) => {
                        e.stopPropagation();
                        this.selectToken(ansIdx, e);
                    };
                }

                span.onmouseenter = (e) => {
                    this.hoverTokenIdx = currentGlobalIdx;
                    if (token.type === 'answer') {
                        this.hoverAnsIdx = token.ans_idx;
                    }
                    this.showValueTooltip(e, currentGlobalIdx);
                    this.updateHighlights();
                    this.updateVision();
                };

                span.onmouseleave = () => {
                    this.hoverAnsIdx = -1;
                    this.hoverTokenIdx = -1;
                    this.hideValueTooltip();
                    this.updateHighlights();
                    this.updateVision();
                };

                content.appendChild(span);
            });

            turnDiv.appendChild(content);
            panel.appendChild(turnDiv);
        });

        panel.onclick = () => this.selectToken(-1);
        this.updateHighlights();
    }

    showValueTooltip(e, tokenIdx) {
        const activeIdx = this.selectedIdx;
        if (activeIdx === -1) return;

        const ansPos = this.data.ans_pos[activeIdx];
        if (tokenIdx >= ansPos) return;

        const relScores = this.getActiveTextRel(activeIdx);
        if (tokenIdx >= relScores.length) return;

        const score = relScores[tokenIdx];
        const tooltip = document.getElementById('value-tooltip');
        tooltip.innerText = `Relevance: ${score.toFixed(4)}`;
        tooltip.style.display = 'block';
        tooltip.style.left = `${e.pageX + 10}px`;
        tooltip.style.top = `${e.pageY + 10}px`;
    }

    hideValueTooltip() {
        document.getElementById('value-tooltip').style.display = 'none';
    }

    clearHoveredCell(blockIdx) {
        delete this.hoveredCells[blockIdx];
        this.updateVision();
    }

    showImageTooltip(e, blockIdx) {
        const activeIdx = this.selectedIdx;
        if (activeIdx === -1) return;

        const blockInfo = this.data.vision_blocks[blockIdx];
        const map = this.getActiveVisionMap(activeIdx, blockIdx);
        if (!blockInfo || !map) return;

        const canvas = e.currentTarget;
        const cols = blockInfo.grid_w;
        const rows = blockInfo.grid_h;
        if (!cols || !rows) return;

        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        const cellW = canvas.width / cols;
        const cellH = canvas.height / rows;
        const col = Math.floor(x / cellW);
        const row = Math.floor(y / cellH);
        if (col < 0 || row < 0 || col >= cols || row >= rows) return;

        const idx = row * cols + col;
        if (idx < 0 || idx >= map.length) return;

        this.hoveredCells[blockIdx] = { row, col };
        this.updateVision();

        const score = map[idx];
        const tooltip = document.getElementById('value-tooltip');
        tooltip.innerText = `Relevance: ${score.toFixed(4)}`;
        tooltip.style.display = 'block';
        tooltip.style.left = `${e.pageX + 10}px`;
        tooltip.style.top = `${e.pageY + 10}px`;
    }

    setupVisibilityTracking() {
        const options = {
            root: document.getElementById('chat-panel'),
            threshold: 0.1
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                const turnIdx = parseInt(entry.target.dataset.turnIdx);
                if (entry.isIntersecting) this.visibleTurns.add(turnIdx);
                else this.visibleTurns.delete(turnIdx);
            });
            this.updateVisionVisibility();
        }, options);

        this.container.querySelectorAll('.tam-turn').forEach(turn => {
            observer.observe(turn);
        });

        this.updateVisionVisibility();
    }

    setupSplitDrag() {
        const splitter = document.getElementById('tam-splitter');
        const chatPanel = document.getElementById('chat-panel');
        const visionPanel = document.getElementById('vision-panel');
        if (!splitter || !chatPanel || !visionPanel) return;

        let isDragging = false;

        const onMouseMove = (e) => {
            if (!isDragging) return;
            const containerRect = this.container.getBoundingClientRect();
            const totalWidth = containerRect.width;
            const offsetX = e.clientX - containerRect.left;
            const splitterWidth = splitter.offsetWidth;
            const minWidth = 200;
            const maxWidth = totalWidth - minWidth - splitterWidth;
            const clamped = Math.min(Math.max(offsetX, minWidth), maxWidth);

            const chatWidth = clamped;
            chatPanel.style.flex = `0 0 ${chatWidth}px`;
            visionPanel.style.flex = `1 1 0px`;
            visionPanel.style.minWidth = `0`;
        };

        const onMouseUp = () => {
            if (!isDragging) return;
            isDragging = false;
            document.body.classList.remove('tam-resizing');
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        };

        splitter.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isDragging = true;
            document.body.classList.add('tam-resizing');
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });
    }

    updateVisionVisibility() {
        this.data.vision_blocks.forEach((block, idx) => {
            const wrapper = document.getElementById(`wrapper-block-${idx}`);
            if (!wrapper) return;

            // Show image if its associated turn is visible
            if (this.visibleTurns.has(block.turn_idx)) {
                wrapper.classList.add('visible');
            } else {
                wrapper.classList.remove('visible');
            }
        });
    }

    selectToken(ansIdx, event = null) {
        this.selectedIdx = ansIdx;
        this.selectedCandidateIdx = 0;
        this.updateCandidateMenu(event);
        this.updateHighlights();
        this.updateVision();
    }

    getActiveVisionMap(ansIdx, blockIdx) {
        let map = this.data.vision_maps?.[ansIdx]?.[blockIdx];
        if (this.selectedIdx === ansIdx && this.selectedCandidateIdx > 0) {
            const candMaps = this.data.candidate_vision_maps?.[ansIdx] || [];
            const candBlockMaps = candMaps[this.selectedCandidateIdx - 1] || [];
            map = candBlockMaps[blockIdx];
        }
        return map;
    }

    getActiveTextRel(ansIdx) {
        let relScores = this.data.text_rel?.[ansIdx] || [];
        if (this.selectedIdx === ansIdx && this.selectedCandidateIdx > 0) {
            const candRel = this.data.candidate_text_rel?.[ansIdx] || [];
            relScores = candRel[this.selectedCandidateIdx - 1] || relScores;
        }
        return relScores;
    }

    setupCandidateMenu() {
        const menu = document.getElementById('candidate-menu');
        if (!menu) return;
        menu.addEventListener('click', (e) => e.stopPropagation());
        this.updateCandidateMenu();
    }

    unpackData() {
        if (this.data.vision_maps) {
            this.data.vision_maps = this.data.vision_maps.map(ansMaps =>
                ansMaps ? ansMaps.map(vMap => this.decodeScores(vMap)) : null
            );
        }
        if (this.data.candidate_vision_maps) {
            this.data.candidate_vision_maps = this.data.candidate_vision_maps.map(ansCandMaps =>
                ansCandMaps ? ansCandMaps.map(candBlockMaps =>
                    candBlockMaps ? candBlockMaps.map(vMap => this.decodeScores(vMap)) : null
                ) : null
            );
        }
        if (this.data.text_rel) {
            this.data.text_rel = this.data.text_rel.map(rel => this.decodeScores(rel));
        }
        if (this.data.candidate_text_rel) {
            this.data.candidate_text_rel = this.data.candidate_text_rel.map(candList =>
                candList ? candList.map(rel => this.decodeScores(rel)) : null
            );
        }
    }

    float16ToFloat32(h) {
        const sign = (h & 0x8000) ? -1 : 1;
        const exp = (h >> 10) & 0x1f;
        const frac = h & 0x03ff;

        if (exp === 0) {
            if (frac === 0) return sign * 0;
            return sign * Math.pow(2, -14) * (frac / 1024);
        }
        if (exp === 31) {
            return frac === 0 ? sign * Infinity : NaN;
        }
        return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
    }

    decodeScores(packed) {
        if (typeof packed !== 'string') return packed;
        if (!packed) return [];

        // New format: float16 raw bytes encoded as base64.
        if (packed.startsWith('f16b64:')) {
            const b64 = packed.slice('f16b64:'.length);
            if (!b64) return new Float32Array(0);
            const bin = atob(b64);
            const pairCount = Math.floor(bin.length / 2);
            const out = new Float32Array(pairCount);
            for (let i = 0; i < pairCount; i++) {
                const lo = bin.charCodeAt(i * 2);
                const hi = bin.charCodeAt(i * 2 + 1);
                const h = lo | (hi << 8);
                out[i] = this.float16ToFloat32(h);
            }
            return out;
        }

        // Backward-compatible fallback: fixed-width scientific notation.
        const res = new Float32Array(packed.length / 8);
        for (let i = 0, j = 0; i < packed.length; i += 8, j++) {
            res[j] = parseFloat(packed.substring(i, i + 8));
        }
        return res;
    }

    updateCandidateMenu(event = null) {
        const menu = document.getElementById('candidate-menu');
        if (!menu) return;
        if (this.selectedIdx === -1) {
            menu.style.display = 'none';
            return;
        }

        const tokens = (this.data.candidate_tokens || [])[this.selectedIdx] || [];
        const logits = (this.data.candidate_logits || [])[this.selectedIdx] || [];
        if (!tokens.length) {
            menu.style.display = 'none';
            return;
        }

        const validLogits = logits.filter(v => typeof v === 'number' && !Number.isNaN(v));
        const minLogit = validLogits.length ? Math.min(0, Math.min(...validLogits)) : 0;
        const maxLogit = validLogits.length ? Math.max(...validLogits) : 1;
        const range = maxLogit - minLogit || 1;

        menu.innerHTML = '';
        tokens.forEach((token, idx) => {
            const item = document.createElement('div');
            item.className = 'tam-candidate-item';
            const label = document.createElement('span');
            label.className = 'tam-candidate-label';
            label.textContent = idx === 0 ? `Answer: ${token}` : `Candidate ${idx}: ${token}`;

            const bar = document.createElement('div');
            bar.className = 'tam-candidate-bar';
            const barFill = document.createElement('div');
            barFill.className = 'tam-candidate-bar-fill';
            const logit = logits[idx];
            if (idx === 0) {
                bar.classList.add('hidden');
            } else if (typeof logit === 'number' && !Number.isNaN(logit)) {
                const norm = (logit - minLogit) / range;
                const pct = 10 + norm * 90;
                barFill.style.width = `${pct}%`;
            } else {
                barFill.style.width = '0%';
            }
            bar.appendChild(barFill);

            item.appendChild(label);
            item.appendChild(bar);
            item.dataset.idx = String(idx);
            if (idx === this.selectedCandidateIdx) {
                item.classList.add('selected');
            }
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectedCandidateIdx = idx;
                this.updateHighlights();
                this.updateVision();
                this.updateCandidateMenu();
            });
            item.addEventListener('mouseenter', (e) => {
                const logitVal = logits[idx];
                if (idx !== 0 && typeof logitVal === 'number' && !Number.isNaN(logitVal)) {
                    const tooltip = document.getElementById('value-tooltip');
                    tooltip.innerText = `Logit: ${logitVal.toFixed(2)}`;
                    tooltip.style.display = 'block';
                    tooltip.style.left = `${e.pageX + 10}px`;
                    tooltip.style.top = `${e.pageY + 10}px`;
                }
            });
            item.addEventListener('mouseleave', () => {
                this.hideValueTooltip();
            });
            menu.appendChild(item);
        });

        menu.style.display = 'block';

        if (event) {
            const panelRect = document.getElementById('chat-panel').getBoundingClientRect();
            const targetRect = event.currentTarget.getBoundingClientRect();
            const rawX = targetRect.left - panelRect.left;
            const rawY = targetRect.bottom - panelRect.top + 4;
            const menuWidth = menu.offsetWidth;
            const menuHeight = menu.offsetHeight;
            const maxX = Math.max(0, panelRect.width - menuWidth);
            const maxY = Math.max(0, panelRect.height - menuHeight);
            const x = Math.min(Math.max(rawX, 0), maxX);
            const y = Math.min(Math.max(rawY, 0), maxY);
            menu.style.left = `${x}px`;
            menu.style.top = `${y}px`;
        }
    }

    highlightImage(visionIdx, isHighlighted) {
        const wrapper = document.getElementById(`wrapper-block-${visionIdx}`);
        if (wrapper) {
            if (isHighlighted) wrapper.classList.add('highlighted');
            else wrapper.classList.remove('highlighted');
        }
    }

    updateHighlights() {
        const tokens = this.container.querySelectorAll('.tam-token');
        const mapSourceIdx = this.selectedIdx !== -1 ? this.selectedIdx : this.hoverAnsIdx;

        tokens.forEach(span => {
            const idx = parseInt(span.dataset.idx);
            span.style.backgroundColor = '';
            span.style.opacity = '1';
            span.classList.remove('selected', 'hovered');

            if (mapSourceIdx !== -1) {
                const ansPos = this.data.ans_pos[mapSourceIdx];
                const relScores = this.getActiveTextRel(mapSourceIdx);

                if (idx < ansPos) {
                    if (idx < relScores.length) {
                        const score = relScores[idx];
                        if (score > 0) {
                            const denom = this.data.max_text_score > 0 ? this.data.max_text_score : 1;
                            span.style.backgroundColor = getJetColor(score / denom);
                        }
                    }
                } else if (idx === ansPos) {
                    if (this.selectedIdx === mapSourceIdx) span.classList.add('selected');
                    else span.classList.add('hovered');
                } else {
                    span.style.opacity = '0.3';
                }
            }
        });
    }

    renderVision() {
        const panel = document.getElementById('vision-panel');
        panel.innerHTML = '';

        this.data.images.forEach((img, idx) => {
            const wrapper = document.createElement('div');
            wrapper.className = 'tam-image-wrapper';
            wrapper.id = `wrapper-block-${idx}`;
            wrapper.innerHTML = `
                <canvas id="canvas-block-${idx}" class="tam-canvas"></canvas>
            `;
            panel.appendChild(wrapper);
            const canvas = wrapper.querySelector('canvas');
            canvas.addEventListener('mousemove', (e) => this.showImageTooltip(e, idx));
            canvas.addEventListener('mouseleave', () => {
                this.hideValueTooltip();
                this.clearHoveredCell(idx);
            });
            this.drawBaseImage(idx);
        });
    }

    drawBaseImage(blockIdx) {
        const canvas = document.getElementById(`canvas-block-${blockIdx}`);
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.src = this.data.images[blockIdx].data_url;
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            this.updateVision();
        };
    }

    updateVision() {
        const mapSourceIdx = this.selectedIdx !== -1 ? this.selectedIdx : this.hoverAnsIdx;

        this.data.images.forEach((img, blockIdx) => {
            const canvas = document.getElementById(`canvas-block-${blockIdx}`);
            if (!canvas) return;
            const ctx = canvas.getContext('2d');

            const baseImg = new Image();
            baseImg.src = this.data.images[blockIdx].data_url;
            ctx.drawImage(baseImg, 0, 0);

            if (mapSourceIdx !== -1) {
                const map = this.getActiveVisionMap(mapSourceIdx, blockIdx);
                const blockInfo = this.data.vision_blocks[blockIdx];
                if (map && blockInfo) {
                    this.overlayHeatmap(canvas, map, blockInfo.grid_h, blockInfo.grid_w, blockIdx);
                }
            }
        });
    }

    overlayHeatmap(canvas, mapData, rows, cols, blockIdx) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        const cellW = width / cols;
        const cellH = height / rows;

        let maxVal = 0;
        for (let i = 0; i < mapData.length; i++) if (mapData[i] > maxVal) maxVal = mapData[i];
        if (maxVal === 0) return;

        for (let i = 0; i < mapData.length; i++) {
            const r = Math.floor(i / cols);
            const c = i % cols;
            const val = mapData[i];
            if (val > 0) {
                ctx.fillStyle = getJetColor(val / maxVal);
                ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
            }
        }

        const hovered = this.hoveredCells[blockIdx];
        if (hovered) {
            const hoverIdx = hovered.row * cols + hovered.col;
            if (hoverIdx >= 0 && hoverIdx < mapData.length && mapData[hoverIdx] > 0) {
                const val = mapData[hoverIdx];
                const scale = 1.15;
                const baseW = cellW;
                const baseH = cellH;
                const drawW = baseW * scale;
                const drawH = baseH * scale;
                const x = hovered.col * cellW - (drawW - baseW) / 2;
                const y = hovered.row * cellH - (drawH - baseH) / 2;
                ctx.save();
                ctx.globalAlpha = 0.9;
                ctx.fillStyle = getJetColor(val / maxVal);
                ctx.fillRect(x, y, drawW + 0.5, drawH + 0.5);
                ctx.restore();
            }
        }
    }
}
