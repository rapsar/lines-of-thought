export function initApp(npyjs) {
    const models = [
        { id: "llama-3.2-1B", name: "llama-3.2-1B" },
        //{ id: "gpt35", name: "GPT-3.5" },
        // Add more models here
    ];

    function populateModelSelector() {
        const modelSelector = document.getElementById("model-selector");
        models.forEach((model) => {
            const option = document.createElement("option");
            option.value = model.id;
            option.textContent = model.name;
            modelSelector.appendChild(option);
        });
        console.log(`Model selector populated with ${models.length} options.`);
    }

    document.addEventListener("DOMContentLoaded", async () => {
        // Select DOM elements
        const plotElement = document.getElementById("plot");
        const vectorSelector = document.getElementById("vector-selector");
        const updateButton = document.getElementById("update-button");
        const timeSelector = document.getElementById("time-selector");
        const excludeLastPointCheckbox = document.getElementById("exclude-last-point");

        // Set default value for vector selector
        vectorSelector.value = "1,2,3";

        let trajectories, singularVectors;

        // Load .npy files
        async function loadNpyFile(url) {
            const npyLoader = new npyjs();
            return await npyLoader.load(url);
        }

        // Load data for a selected model
        async function loadDataForModel(modelId) {
            try {
                const trajectories = await loadNpyFile(`assets/${modelId}/trajectories.npy`);
                const singularVectors = await loadNpyFile(`assets/${modelId}/singular_vectors.npy`);

                console.log(`Data loaded for model: ${modelId}`);
                console.log(`Trajectories shape: ${trajectories.shape}`);
                console.log(`Singular vectors shape: ${singularVectors.shape}`);

                populateTimeSelector(singularVectors.shape[2]);
                return { trajectories, singularVectors };
            } catch (error) {
                console.error(`Error loading data for model ${modelId}:`, error);
                alert("Failed to load data. Check the console for details.");
                return null;
            }
        }

        // Populate the time selector dropdown
        function populateTimeSelector(numTimes) {
            timeSelector.innerHTML = "";
            for (let i = 0; i < numTimes; i++) {
                const option = document.createElement("option");
                option.value = i; // Zero-based index
                option.textContent = i + 1; // One-based display
                timeSelector.appendChild(option);
            }
            console.log(`Time selector populated with ${numTimes} options.`);
        }

        // Project trajectories onto singular vectors
        function projectData(selectedTime, selectedLayers, excludeLastPoint) {
            if (!trajectories || !singularVectors) {
                console.error("Data not loaded.");
                return [];
            }

            const numDimensions = singularVectors.shape[0];
            if (selectedTime < 0 || selectedTime >= singularVectors.shape[2]) {
                console.error("Selected time is out of bounds.");
                return [];
            }

            const selectedVectors = selectedLayers.map((layer) => {
                const timeOffset = selectedTime * numDimensions * numDimensions;
                const layerStart = timeOffset + layer * numDimensions;
                const layerEnd = layerStart + numDimensions;
                return singularVectors.data.slice(layerStart, layerEnd);
            });

            const projectedTrajectories = [];
            const numTrajectories = Math.min(trajectories.shape[2], 100);
            for (let t = 0; t < numTrajectories; t++) {
                const trajectory = [];
                const maxLength = excludeLastPoint ? trajectories.shape[1] - 1 : trajectories.shape[1];
                for (let l = 0; l < maxLength; l++) {
                    const point = selectedVectors.map((vector) => {
                        return vector.reduce(
                            (sum, value, i) =>
                                sum +
                                value *
                                    trajectories.data[
                                        t * trajectories.shape[1] * trajectories.shape[0] +
                                            l * trajectories.shape[0] +
                                            i
                                    ],
                            0
                        );
                    });
                    trajectory.push(point);
                }
                projectedTrajectories.push(trajectory);
            }

            return projectedTrajectories;
        }

        // Prepare data for Plotly visualization
        function preparePlotData(projectedData) {
            return projectedData.map((trajectory, idx) => {
                const numPoints = trajectory.length;
                const colors = Array.from({ length: numPoints }, (_, i) => {
                    const t = i / (numPoints - 1);
                    return `rgb(${Math.round(255 * t)}, 0, ${Math.round(255 * (1 - t))})`;
                });

                return {
                    x: trajectory.map((point) => point[0]),
                    y: trajectory.map((point) => point[1]),
                    z: trajectory.map((point) => point[2]),
                    mode: "lines",
                    type: "scatter3d",
                    line: {
                        color: colors,
                        width: 2,
                    },
                    name: `Trajectory ${idx + 1}`,
                };
            });
        }

        // Update the Plotly visualization
        function updatePlot(selectedTime, selectedLayers) {
            const excludeLastPoint = excludeLastPointCheckbox.checked; // Check the checkbox state
            console.log(
                `Updating plot: time=${selectedTime}, layers=${selectedLayers}, excludeLastPoint=${excludeLastPoint}`
            );
            try {
                const projectedData = projectData(selectedTime, selectedLayers, excludeLastPoint);
                if (!projectedData.length) throw new Error("No data to plot.");

                const plotData = preparePlotData(projectedData);
                const layout = {
                    title: "Trajectories in Latent Space",
                    showlegend: false,
                    scene: {
                        xaxis: { title: "SVD1" },
                        yaxis: { title: "SVD2" },
                        zaxis: { title: "SVD3" },
                    },
                };

                Plotly.newPlot(plotElement, plotData, layout).then(() => {
                    Plotly.relayout(plotElement, {
                        "scene.hovermode": false,
                        "scene.xaxis.showspikes": false,
                        "scene.yaxis.showspikes": false,
                        "scene.zaxis.showspikes": false,
                    });
                });
            } catch (error) {
                console.error("Error updating plot:", error.message);
                alert("Failed to update the plot. Check the console for details.");
            }
        }

        // Parse user input for selected layers
        function parseInput(input) {
            const totalLayers = singularVectors.shape[1];
            return input.split(",").map((str) => {
                str = str.trim();
                if (str.startsWith("last")) {
                    return totalLayers - parseInt(str.split("-")[1]);
                } else {
                    return parseInt(str) - 1;
                }
            });
        }

        // Populate the model selector
        populateModelSelector();

        // Event listener for model selection
        const modelSelector = document.getElementById("model-selector");
        modelSelector.addEventListener("change", async (event) => {
            const selectedModel = event.target.value;
            const data = await loadDataForModel(selectedModel);
            if (data) {
                trajectories = data.trajectories;
                singularVectors = data.singularVectors;
                updatePlot(0, [0, 1, 2]); // Default plot
            }
        });

        // Event listeners for other interactions
        timeSelector.addEventListener("change", () => {
            const selectedTime = parseInt(timeSelector.value, 10);
            const selectedLayers = parseInput(vectorSelector.value);
            if (selectedLayers.length) updatePlot(selectedTime, selectedLayers);
        });

        updateButton.addEventListener("click", () => {
            const selectedTime = parseInt(timeSelector.value, 10);
            const selectedLayers = parseInput(vectorSelector.value);
            if (selectedLayers.length) updatePlot(selectedTime, selectedLayers);
        });

        excludeLastPointCheckbox.addEventListener("change", () => {
            const selectedTime = parseInt(timeSelector.value, 10);
            const selectedLayers = parseInput(vectorSelector.value);
            if (selectedLayers.length) updatePlot(selectedTime, selectedLayers);
        });

        // Load and display the default model
        const defaultModel = models[0].id;
        const data = await loadDataForModel(defaultModel);
        if (data) {
            trajectories = data.trajectories;
            singularVectors = data.singularVectors;
            updatePlot(0, [0, 1, 2]); // Default plot
        }
    });
}