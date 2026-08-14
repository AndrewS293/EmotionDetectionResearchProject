const fileInput = document.getElementById("jsonFile");
const fileName = document.getElementById("fileName");

const results = document.getElementById("results");
const errorBox = document.getElementById("error");

fileInput.addEventListener("change", handleFile);


function handleFile(event) {

    const file = event.target.files[0];

    if (!file) {
        return;
    }

    fileName.textContent = file.name;

    const reader = new FileReader();

    reader.onload = function(e) {

        try {

            const data = JSON.parse(e.target.result);

            displayReasoning(data);

        } catch (error) {

            showError("Unable to read this file as valid JSON.");

        }

    };

    reader.readAsText(file);
}


function displayReasoning(data) {

    errorBox.classList.add("hidden");

    /*
     * Your API currently returns reasoning as a JSON string.
     *
     * Example:
     *
     * "reasoning": "{ ... }"
     *
     * So we parse it here.
     */

    let reasoning = data.reasoning;

    if (typeof reasoning === "string") {
        try {
            reasoning = JSON.parse(reasoning);
        } catch (error) {
            showError("The reasoning field contains invalid JSON.");
            return;
        }
    }


    // Make sure reasoning exists

    if (!reasoning) {
        showError("No reasoning data was found in this JSON file.");
        return;
    }


    // Detected state

    document.getElementById("detectedState").textContent =
        reasoning.detected_state || "--";


    // Confidence

    const confidence = reasoning.confidence_percent;

    document.getElementById("confidence").textContent =
        confidence !== undefined
            ? `${confidence}%`
            : "--%";


    // Model agreement

    const agreement = reasoning.model_agreement || {};

    document.getElementById("agreementSummary").textContent =
        agreement.summary || "No model agreement information available.";


    // Agreeing models

    const agreeingList =
        document.getElementById("agreeingModels");

    agreeingList.innerHTML = "";

    const agreeingModels =
        agreement.agreeing_models || [];

    agreeingModels.forEach(model => {

        const li = document.createElement("li");

        li.className = "agree";
        li.textContent = `✓ ${model}`;

        agreeingList.appendChild(li);

    });


    // Disagreeing models

    const disagreeingList =
        document.getElementById("disagreeingModels");

    disagreeingList.innerHTML = "";

    const disagreeingModels =
        agreement.disagreeing_models || [];

    disagreeingModels.forEach(model => {

        const li = document.createElement("li");

        li.className = "disagree";
        li.textContent = `✗ ${model}`;

        disagreeingList.appendChild(li);

    });


    // Physiological interpretation

    document.getElementById(
        "physiologicalInterpretation"
    ).textContent =
        reasoning.physiological_interpretation ||
        "No physiological interpretation available.";


    // Trend

    document.getElementById("trend").textContent =
        reasoning.trend ||
        "No trend information available.";


    // User summary

    document.getElementById("userSummary").textContent =
        reasoning.user_summary ||
        "No summary available.";


    // Show results

    results.classList.remove("hidden");
}


function showError(message) {

    errorBox.textContent = message;
    errorBox.classList.remove("hidden");

    results.classList.add("hidden");
}