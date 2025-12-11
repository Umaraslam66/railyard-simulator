// Simulation Controller
class RailyardSimulation {
    constructor() {
        this.isRunning = false;
        this.trains = [];
        this.speed = 1;
        this.density = 5;
        this.animationId = null;
        this.pathCache = new Map(); // Cache calculated paths
        this.lastTime = 0;
        
        this.init();
    }
    
    init() {
        console.log('Initializing railyard simulation...');
        this.drawInfrastructure();
        this.setupControls();
        this.updateStats();
        console.log('Initialization complete');
    }
    
    // Draw railway infrastructure with enhanced visuals
    drawInfrastructure() {
        const tracksGroup = document.getElementById('tracks');
        const switchesGroup = document.getElementById('switches');
        
        // Draw all tracks with enhanced styling
        RAILYARD_DATA.edges.forEach(([node1, node2, length]) => {
            const coord1 = RAILYARD_DATA.coordinates[node1];
            const coord2 = RAILYARD_DATA.coordinates[node2];
            
            if (!coord1 || !coord2) return;
            
            const trackType = this.getTrackType(node1, node2);
            const path = this.createTrackPath(coord1, coord2, trackType);
            
            // Create track group for multiple layers
            const trackGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            trackGroup.setAttribute('class', 'track-group');
            
            // Draw railroad ties (sleepers) first
            this.drawRailTies(trackGroup, coord1, coord2, trackType);
            
            // Base track (darker, wider)
            const baseLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            baseLine.setAttribute('d', path);
            baseLine.setAttribute('class', 'track track-base');
            baseLine.setAttribute('stroke', this.darkenColor(trackType.color, 20));
            baseLine.setAttribute('stroke-width', '6');
            baseLine.setAttribute('fill', 'none');
            baseLine.setAttribute('opacity', '0.5');
            trackGroup.appendChild(baseLine);
            
            // Main track line (metallic effect with gradient)
            const trackLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            trackLine.setAttribute('d', path);
            trackLine.setAttribute('class', 'track');
            trackLine.setAttribute('stroke', trackType.color);
            trackLine.setAttribute('stroke-width', '4');
            trackLine.setAttribute('fill', 'none');
            trackLine.setAttribute('data-from', node1);
            trackLine.setAttribute('data-to', node2);
            trackLine.setAttribute('data-length', length);
            
            // Add metallic shine effect
            trackLine.style.strokeLinecap = 'round';
            trackLine.style.strokeLinejoin = 'round';
            
            trackGroup.appendChild(trackLine);
            
            // Add hover effect
            trackGroup.addEventListener('mouseenter', (e) => {
                this.showTooltip(e, `Track ${node1} → ${node2}<br>Length: ${length}m<br>Type: ${trackType.type}`);
                trackLine.style.strokeWidth = '7';
                trackLine.style.filter = 'drop-shadow(0 0 8px ' + trackType.color + ')';
            });
            trackGroup.addEventListener('mouseleave', () => {
                this.hideTooltip();
                trackLine.style.strokeWidth = '4';
                trackLine.style.filter = '';
            });
            
            tracksGroup.appendChild(trackGroup);
        });
        
        // Draw all switches with enhanced 3D effect
        RAILYARD_DATA.switches.forEach(switchId => {
            const coord = RAILYARD_DATA.coordinates[switchId];
            if (!coord) return;
            
            const switchGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            switchGroup.setAttribute('class', 'switch');
            switchGroup.setAttribute('data-id', switchId);
            
            // Switch shadow
            const shadowCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            shadowCircle.setAttribute('cx', coord[0] + 2);
            shadowCircle.setAttribute('cy', coord[1] + 2);
            shadowCircle.setAttribute('r', '10');
            shadowCircle.setAttribute('fill', 'rgba(0, 0, 0, 0.3)');
            shadowCircle.setAttribute('filter', 'blur(2px)');
            switchGroup.appendChild(shadowCircle);
            
            // Outer glow circle
            const glowCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            glowCircle.setAttribute('cx', coord[0]);
            glowCircle.setAttribute('cy', coord[1]);
            glowCircle.setAttribute('r', '12');
            glowCircle.setAttribute('fill', '#2980b9');
            glowCircle.setAttribute('opacity', '0.3');
            switchGroup.appendChild(glowCircle);
            
            // Main switch circle with gradient
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', coord[0]);
            circle.setAttribute('cy', coord[1]);
            circle.setAttribute('r', '8');
            circle.setAttribute('fill', '#3498db');
            circle.setAttribute('stroke', '#2980b9');
            circle.setAttribute('stroke-width', '2.5');
            switchGroup.appendChild(circle);
            
            // Inner highlight
            const highlight = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            highlight.setAttribute('cx', coord[0] - 2);
            highlight.setAttribute('cy', coord[1] - 2);
            highlight.setAttribute('r', '3');
            highlight.setAttribute('fill', '#5dade2');
            highlight.setAttribute('opacity', '0.7');
            switchGroup.appendChild(highlight);
            
            // Switch label with background
            const labelBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            labelBg.setAttribute('x', coord[0] - 15);
            labelBg.setAttribute('y', coord[1] - 28);
            labelBg.setAttribute('width', '30');
            labelBg.setAttribute('height', '16');
            labelBg.setAttribute('rx', '3');
            labelBg.setAttribute('fill', 'rgba(255, 255, 255, 0.9)');
            labelBg.setAttribute('stroke', '#3498db');
            labelBg.setAttribute('stroke-width', '1');
            switchGroup.appendChild(labelBg);
            
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', coord[0]);
            text.setAttribute('y', coord[1] - 17);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('font-size', '12');
            text.setAttribute('font-weight', 'bold');
            text.setAttribute('fill', '#2c3e50');
            text.textContent = switchId;
            switchGroup.appendChild(text);
            
            // Add hover effect
            switchGroup.addEventListener('mouseenter', (e) => {
                this.showTooltip(e, `Switch ${switchId}<br>Node in railway network`);
                circle.setAttribute('r', '12');
                glowCircle.setAttribute('r', '16');
                glowCircle.setAttribute('opacity', '0.5');
            });
            switchGroup.addEventListener('mouseleave', () => {
                this.hideTooltip();
                circle.setAttribute('r', '8');
                glowCircle.setAttribute('r', '12');
                glowCircle.setAttribute('opacity', '0.3');
            });
            
            switchesGroup.appendChild(switchGroup);
        });
    }
    
    // Draw railroad ties (sleepers) along the track
    drawRailTies(parentGroup, coord1, coord2, trackType) {
        const [x1, y1] = coord1;
        const [x2, y2] = coord2;
        const distance = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
        const numTies = Math.floor(distance / 50); // One tie every 50 units
        
        for (let i = 0; i <= numTies; i++) {
            const t = i / numTies;
            const position = this.getPointOnPath(coord1, coord2, trackType, t);
            
            if (position) {
                const tie = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                const tieLength = 12;
                const angle = (position.angle + 90) * Math.PI / 180; // Perpendicular to track
                
                const x1tie = position.x + Math.cos(angle) * tieLength;
                const y1tie = position.y + Math.sin(angle) * tieLength;
                const x2tie = position.x - Math.cos(angle) * tieLength;
                const y2tie = position.y - Math.sin(angle) * tieLength;
                
                tie.setAttribute('x1', x1tie);
                tie.setAttribute('y1', y1tie);
                tie.setAttribute('x2', x2tie);
                tie.setAttribute('y2', y2tie);
                tie.setAttribute('stroke', '#8B4513');
                tie.setAttribute('stroke-width', '3');
                tie.setAttribute('stroke-linecap', 'round');
                tie.setAttribute('opacity', '0.6');
                
                parentGroup.appendChild(tie);
            }
        }
    }
    
    // Determine track type for styling
    getTrackType(node1, node2) {
        // Check polyline tracks
        for (let track of RAILYARD_DATA.trackTypes.polyline) {
            if ((track.edge[0] === node1 && track.edge[1] === node2) ||
                (track.edge[0] === node2 && track.edge[1] === node1)) {
                return { type: 'polyline', color: '#3498db', data: track };
            }
        }
        
        // Check curved tracks
        for (let track of RAILYARD_DATA.trackTypes.curved) {
            if ((track.edge[0] === node1 && track.edge[1] === node2) ||
                (track.edge[0] === node2 && track.edge[1] === node1)) {
                return { type: 'curved', color: '#e74c3c', data: track };
            }
        }
        
        // Check mixed tracks
        for (let track of RAILYARD_DATA.trackTypes.mixed) {
            if ((track.edge[0] === node1 && track.edge[1] === node2) ||
                (track.edge[0] === node2 && track.edge[1] === node1)) {
                return { type: 'mixed', color: '#27ae60', data: track };
            }
        }
        
        // Default straight track
        return { type: 'straight', color: '#2c3e50' };
    }
    
    // Create SVG path for track - EXACT replication of Python version
    createTrackPath(coord1, coord2, trackType) {
        const [x1, y1] = coord1;
        const [x2, y2] = coord2;
        
        if (trackType.type === 'curved') {
            // Quadratic Bézier curve
            const controlX = (x1 + x2) / 2 + (trackType.data?.controlOffset[0] || 0);
            const controlY = (y1 + y2) / 2 + (trackType.data?.controlOffset[1] || 0);
            return `M ${x1} ${y1} Q ${controlX} ${controlY}, ${x2} ${y2}`;
        } else if (trackType.type === 'polyline') {
            // Polyline with alternating offsets
            const segments = trackType.data?.segments || 2;
            const offset = trackType.data?.offset || 0;
            let path = `M ${x1} ${y1}`;
            
            for (let i = 1; i < segments; i++) {
                const t = i / segments;
                let midX = (1 - t) * x1 + t * x2;
                let midY = (1 - t) * y1 + t * y2;
                
                // Perpendicular direction for offset (90-degree rotation)
                const dx = x2 - x1;
                const dy = y2 - y1;
                const length = Math.sqrt(dx * dx + dy * dy);
                const perpX = -dy / length;
                const perpY = dx / length;
                
                // Apply offset in perpendicular direction
                midX += perpX * offset * Math.pow(-1, i);
                midY += perpY * offset * Math.pow(-1, i);
                
                path += ` L ${midX} ${midY}`;
            }
            
            path += ` L ${x2} ${y2}`;
            return path;
        } else if (trackType.type === 'mixed') {
            // Mixed edge with straight and curve
            const data = trackType.data;
            const angle = data?.angle || 0;
            const straightLen = data?.straightLength || 100;
            const controlOffset = data?.controlOffset || [0, 0];
            
            if (data?.curveFirst) {
                // Curve followed by straight line
                const controlX = x1 + controlOffset[0];
                const controlY = y1 + controlOffset[1];
                const intersectX = x2 - straightLen * Math.cos(angle);
                const intersectY = y2 - straightLen * Math.sin(angle);
                return `M ${x1} ${y1} Q ${controlX} ${controlY}, ${intersectX} ${intersectY} L ${x2} ${y2}`;
            } else {
                // Straight line followed by curve
                const straightX = x1 + straightLen * Math.cos(angle);
                const straightY = y1 + straightLen * Math.sin(angle);
                const controlX = (straightX + x2) / 2 + controlOffset[0];
                const controlY = (straightY + y2) / 2 + controlOffset[1];
                return `M ${x1} ${y1} L ${straightX} ${straightY} Q ${controlX} ${controlY}, ${x2} ${y2}`;
            }
        }
        
        // Default straight line
        return `M ${x1} ${y1} L ${x2} ${y2}`;
    }
    
    // Calculate exact point on path using same logic as visualization
    getPointOnPath(coord1, coord2, trackType, progress) {
        const [x1, y1] = coord1;
        const [x2, y2] = coord2;
        const t = Math.max(0, Math.min(1, progress)); // Clamp to [0, 1]
        
        if (trackType.type === 'curved') {
            // Quadratic Bézier interpolation
            const controlX = (x1 + x2) / 2 + (trackType.data?.controlOffset[0] || 0);
            const controlY = (y1 + y2) / 2 + (trackType.data?.controlOffset[1] || 0);
            const x = (1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * controlX + t * t * x2;
            const y = (1 - t) * (1 - t) * y1 + 2 * (1 - t) * t * controlY + t * t * y2;
            
            // Calculate tangent for rotation
            const tx = 2 * (1 - t) * (controlX - x1) + 2 * t * (x2 - controlX);
            const ty = 2 * (1 - t) * (controlY - y1) + 2 * t * (y2 - controlY);
            const angle = Math.atan2(ty, tx) * 180 / Math.PI;
            
            return { x, y, angle };
        } else if (trackType.type === 'polyline') {
            // Polyline interpolation
            const segments = trackType.data?.segments || 2;
            const offset = trackType.data?.offset || 0;
            
            // Find which segment we're in
            const segmentIndex = Math.floor(t * segments);
            const segmentProgress = (t * segments) - segmentIndex;
            
            // Calculate start and end points of current segment
            let startX, startY, endX, endY;
            
            if (segmentIndex === 0) {
                startX = x1;
                startY = y1;
            } else {
                const st = segmentIndex / segments;
                startX = (1 - st) * x1 + st * x2;
                startY = (1 - st) * y1 + st * y2;
                
                const dx = x2 - x1;
                const dy = y2 - y1;
                const length = Math.sqrt(dx * dx + dy * dy);
                const perpX = -dy / length;
                const perpY = dx / length;
                startX += perpX * offset * Math.pow(-1, segmentIndex);
                startY += perpY * offset * Math.pow(-1, segmentIndex);
            }
            
            if (segmentIndex >= segments - 1) {
                endX = x2;
                endY = y2;
            } else {
                const et = (segmentIndex + 1) / segments;
                endX = (1 - et) * x1 + et * x2;
                endY = (1 - et) * y1 + et * y2;
                
                const dx = x2 - x1;
                const dy = y2 - y1;
                const length = Math.sqrt(dx * dx + dy * dy);
                const perpX = -dy / length;
                const perpY = dx / length;
                endX += perpX * offset * Math.pow(-1, segmentIndex + 1);
                endY += perpY * offset * Math.pow(-1, segmentIndex + 1);
            }
            
            const x = startX + segmentProgress * (endX - startX);
            const y = startY + segmentProgress * (endY - startY);
            const angle = Math.atan2(endY - startY, endX - startX) * 180 / Math.PI;
            
            return { x, y, angle };
        } else if (trackType.type === 'mixed') {
            // Mixed edge interpolation
            const data = trackType.data;
            const angle_rad = data?.angle || 0;
            const straightLen = data?.straightLength || 100;
            const controlOffset = data?.controlOffset || [0, 0];
            
            const totalDist = Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
            const straightRatio = straightLen / totalDist;
            
            if (data?.curveFirst) {
                // Curve first
                const controlX = x1 + controlOffset[0];
                const controlY = y1 + controlOffset[1];
                const intersectX = x2 - straightLen * Math.cos(angle_rad);
                const intersectY = y2 - straightLen * Math.sin(angle_rad);
                
                const curveRatio = 1 - straightRatio;
                
                if (t < curveRatio) {
                    // On curve
                    const ct = t / curveRatio;
                    const x = (1 - ct) * (1 - ct) * x1 + 2 * (1 - ct) * ct * controlX + ct * ct * intersectX;
                    const y = (1 - ct) * (1 - ct) * y1 + 2 * (1 - ct) * ct * controlY + ct * ct * intersectY;
                    const tx = 2 * (1 - ct) * (controlX - x1) + 2 * ct * (intersectX - controlX);
                    const ty = 2 * (1 - ct) * (controlY - y1) + 2 * ct * (intersectY - controlY);
                    const angle = Math.atan2(ty, tx) * 180 / Math.PI;
                    return { x, y, angle };
                } else {
                    // On straight
                    const st = (t - curveRatio) / straightRatio;
                    const x = intersectX + st * (x2 - intersectX);
                    const y = intersectY + st * (y2 - intersectY);
                    const angle = Math.atan2(y2 - intersectY, x2 - intersectX) * 180 / Math.PI;
                    return { x, y, angle };
                }
            } else {
                // Straight first
                const straightX = x1 + straightLen * Math.cos(angle_rad);
                const straightY = y1 + straightLen * Math.sin(angle_rad);
                
                if (t < straightRatio) {
                    // On straight
                    const st = t / straightRatio;
                    const x = x1 + st * (straightX - x1);
                    const y = y1 + st * (straightY - y1);
                    const angle = Math.atan2(straightY - y1, straightX - x1) * 180 / Math.PI;
                    return { x, y, angle };
                } else {
                    // On curve
                    const ct = (t - straightRatio) / (1 - straightRatio);
                    const controlX = (straightX + x2) / 2 + controlOffset[0];
                    const controlY = (straightY + y2) / 2 + controlOffset[1];
                    const x = (1 - ct) * (1 - ct) * straightX + 2 * (1 - ct) * ct * controlX + ct * ct * x2;
                    const y = (1 - ct) * (1 - ct) * straightY + 2 * (1 - ct) * ct * controlY + ct * ct * y2;
                    const tx = 2 * (1 - ct) * (controlX - straightX) + 2 * ct * (x2 - controlX);
                    const ty = 2 * (1 - ct) * (controlY - straightY) + 2 * ct * (y2 - controlY);
                    const angle = Math.atan2(ty, tx) * 180 / Math.PI;
                    return { x, y, angle };
                }
            }
        }
        
        // Default straight line interpolation
        const x = x1 + t * (x2 - x1);
        const y = y1 + t * (y2 - y1);
        const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
        return { x, y, angle };
    }
    
    // Setup control buttons and sliders
    setupControls() {
        document.getElementById('playBtn').addEventListener('click', () => this.start());
        document.getElementById('pauseBtn').addEventListener('click', () => this.pause());
        document.getElementById('resetBtn').addEventListener('click', () => this.reset());
        
        document.getElementById('trafficDensity').addEventListener('input', (e) => {
            this.density = parseInt(e.target.value);
            document.getElementById('densityValue').textContent = `${this.density}`;
            if (this.isRunning) {
                this.reset();
                this.start();
            }
        });
        
        document.getElementById('speedControl').addEventListener('input', (e) => {
            this.speed = parseFloat(e.target.value);
            document.getElementById('speedValue').textContent = `${this.speed}x`;
        });
    }
    
    // Start simulation
    start() {
        if (this.isRunning) return;
        
        console.log('Starting simulation...');
        this.isRunning = true;
        this.lastTime = 0;
        this.createTrains();
        
        console.log(`Animation starting with ${this.trains.length} trains`);
        
        // Use arrow function to preserve 'this' context
        const animateLoop = (time) => {
            if (!this.isRunning) return;
            
            if (!this.lastTime) this.lastTime = time;
            const deltaTime = Math.min((time - this.lastTime) / 1000, 0.1); // Cap at 100ms
            this.lastTime = time;
            
            let activeTrains = 0;
            
            this.trains.forEach((train, index) => {
                if (time < train.delay) return;
                
                activeTrains++;
                
                // Simple linear progress
                train.progress += 0.002 * this.speed * train.route.speed;
                
                if (train.progress >= 1) {
                    train.progress = 0;
                    train.pathIndex++;
                    
                    if (train.pathIndex >= train.route.path.length - 1) {
                        train.pathIndex = 0;
                    }
                }
                
                const fromNode = train.route.path[train.pathIndex];
                const toNode = train.route.path[train.pathIndex + 1];
                
                if (fromNode !== undefined && toNode !== undefined) {
                    const from = RAILYARD_DATA.coordinates[fromNode];
                    const to = RAILYARD_DATA.coordinates[toNode];
                    
                    if (from && to) {
                        // Simple linear interpolation
                        const x = from[0] + (to[0] - from[0]) * train.progress;
                        const y = from[1] + (to[1] - from[1]) * train.progress;
                        const angle = Math.atan2(to[1] - from[1], to[0] - from[0]) * 180 / Math.PI;
                        
                        train.element.setAttribute('transform', `translate(${x}, ${y}) rotate(${angle})`);
                        train.element.style.display = 'block';
                        
                        if (train.progress < 0.1) {
                            this.highlightSwitch(fromNode);
                        }
                    }
                }
            });
            
            this.animationId = requestAnimationFrame(animateLoop);
        };
        
        this.animationId = requestAnimationFrame(animateLoop);
        
        document.getElementById('playBtn').style.opacity = '0.5';
        document.getElementById('pauseBtn').style.opacity = '1';
        
        console.log('Animation loop started');
    }
    
    // Pause simulation
    pause() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        document.getElementById('playBtn').style.opacity = '1';
        document.getElementById('pauseBtn').style.opacity = '0.5';
    }
    
    // Reset simulation
    reset() {
        this.pause();
        this.trains = [];
        document.getElementById('trains').innerHTML = '';
        this.updateStats();
        
        document.getElementById('playBtn').style.opacity = '1';
        document.getElementById('pauseBtn').style.opacity = '0.5';
    }
    
    // Create trains based on density
    createTrains() {
        const trainsGroup = document.getElementById('trains');
        trainsGroup.innerHTML = '';
        this.trains = [];
        
        const currentTime = Date.now();
        
        for (let i = 0; i < this.density; i++) {
            const route = TRAIN_ROUTES[i % TRAIN_ROUTES.length];
            
            // Validate route has valid path
            if (!route.path || route.path.length < 2) {
                console.error(`Invalid route for train ${i}:`, route);
                continue;
            }
            
            const train = this.createTrain(route, i, currentTime);
            this.trains.push(train);
        }
        
        console.log(`Created ${this.trains.length} trains successfully`);
        this.updateStats();
    }
    
    // Create individual train with detailed 3D-like design
    createTrain(route, index, currentTime) {
        const trainGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        trainGroup.setAttribute('class', 'train');
        trainGroup.setAttribute('data-id', route.id);
        
        // Create gradient for train body
        const gradientId = `trainGradient${route.id}_${index}`;
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
        gradient.setAttribute('id', gradientId);
        gradient.setAttribute('x1', '0%');
        gradient.setAttribute('y1', '0%');
        gradient.setAttribute('x2', '0%');
        gradient.setAttribute('y2', '100%');
        
        const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop1.setAttribute('offset', '0%');
        stop1.setAttribute('style', `stop-color:${route.color};stop-opacity:1`);
        
        const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        stop2.setAttribute('offset', '100%');
        stop2.setAttribute('style', `stop-color:${this.darkenColor(route.color, 30)};stop-opacity:1`);
        
        gradient.appendChild(stop1);
        gradient.appendChild(stop2);
        defs.appendChild(gradient);
        trainGroup.appendChild(defs);
        
        // Main train body
        const body = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        body.setAttribute('x', '-18');
        body.setAttribute('y', '-10');
        body.setAttribute('width', '36');
        body.setAttribute('height', '20');
        body.setAttribute('rx', '4');
        body.setAttribute('fill', `url(#${gradientId})`);
        body.setAttribute('stroke', this.darkenColor(route.color, 40));
        body.setAttribute('stroke-width', '2');
        trainGroup.appendChild(body);
        
        // Front nose
        const nose = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        nose.setAttribute('d', 'M 18 -8 L 25 0 L 18 8 Z');
        nose.setAttribute('fill', route.color);
        nose.setAttribute('stroke', this.darkenColor(route.color, 40));
        nose.setAttribute('stroke-width', '2');
        trainGroup.appendChild(nose);
        
        // Windows
        for (let i = 0; i < 4; i++) {
            const window = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            window.setAttribute('x', -14 + i * 8);
            window.setAttribute('y', '-6');
            window.setAttribute('width', '5');
            window.setAttribute('height', '7');
            window.setAttribute('rx', '1');
            window.setAttribute('fill', '#87CEEB');
            window.setAttribute('stroke', '#4682B4');
            window.setAttribute('stroke-width', '0.5');
            window.setAttribute('opacity', '0.8');
            trainGroup.appendChild(window);
        }
        
        // Wheels
        for (let i = 0; i < 4; i++) {
            const wheelX = -12 + i * 8;
            const wheel = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            wheel.setAttribute('cx', wheelX);
            wheel.setAttribute('cy', '10');
            wheel.setAttribute('r', '3');
            wheel.setAttribute('fill', '#2c3e50');
            wheel.setAttribute('stroke', '#1a252f');
            wheel.setAttribute('stroke-width', '1');
            trainGroup.appendChild(wheel);
        }
        
        // Headlight
        const headlight = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        headlight.setAttribute('cx', '22');
        headlight.setAttribute('cy', '0');
        headlight.setAttribute('r', '2');
        headlight.setAttribute('fill', '#FFD700');
        trainGroup.appendChild(headlight);
        
        document.getElementById('trains').appendChild(trainGroup);
        
        // Set initial position
        const startNode = route.path[0];
        if (startNode !== undefined && RAILYARD_DATA.coordinates[startNode]) {
            const coord = RAILYARD_DATA.coordinates[startNode];
            trainGroup.setAttribute('transform', `translate(${coord[0]}, ${coord[1]}) rotate(0)`);
            trainGroup.style.opacity = '1';
        }
        
        console.log(`Train ${index} created for route ${route.name}, starting at node ${route.path[0]}`);
        
        return {
            element: trainGroup,
            route: route,
            pathIndex: 0,
            progress: 0,
            delay: currentTime + (index * 2000)
        };
    }
    
    // Helper function to darken color
    darkenColor(color, percent) {
        const num = parseInt(color.replace("#", ""), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) - amt;
        const G = (num >> 8 & 0x00FF) - amt;
        const B = (num & 0x0000FF) - amt;
        return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255))
            .toString(16).slice(1);
    }
    
    // Helper function to lighten color
    lightenColor(color, percent) {
        const num = parseInt(color.replace("#", ""), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) + amt;
        const G = (num >> 8 & 0x00FF) + amt;
        const B = (num & 0x0000FF) + amt;
        return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255))
            .toString(16).slice(1);
    }
    
    
    // Highlight switch when train passes through
    highlightSwitch(switchId) {
        const switchElement = document.querySelector(`.switch[data-id="${switchId}"]`);
        if (switchElement) {
            switchElement.classList.add('active');
            setTimeout(() => {
                switchElement.classList.remove('active');
            }, 500);
        }
    }
    
    // Update statistics
    updateStats() {
        document.getElementById('activeTrains').textContent = this.trains.length;
    }
    
    // Tooltip functions
    showTooltip(event, text) {
        let tooltip = document.querySelector('.tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            document.body.appendChild(tooltip);
        }
        
        tooltip.innerHTML = text;
        tooltip.style.left = event.pageX + 15 + 'px';
        tooltip.style.top = event.pageY + 15 + 'px';
        tooltip.classList.add('show');
    }
    
    hideTooltip() {
        const tooltip = document.querySelector('.tooltip');
        if (tooltip) {
            tooltip.classList.remove('show');
        }
    }
}

// Initialize simulation when page loads
document.addEventListener('DOMContentLoaded', () => {
    const simulation = new RailyardSimulation();
});
