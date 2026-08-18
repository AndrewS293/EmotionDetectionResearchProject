const API_URL = "http://127.0.0.1:8000/predict";


const fileInput = document.getElementById("jsonFile");
const sendButton = document.getElementById("sendButton");

const fileName = document.getElementById("fileName");
const status = document.getElementById("status");

const results = document.getElementById("results");

let selectedFile = null;


/* ============================================
   FILE SELECTION
============================================ */

fileInput.addEventListener("change", function(event) {

    selectedFile = event.target.files[0];

    if (!selectedFile) {

        sendButton.disabled = true;

        fileName.textContent = "No file selected";

        return;
    }

    fileName.textContent = selectedFile.name;

    sendButton.disabled = false;

    status.textContent = "";

});


/* ============================================
   SEND TO API
============================================ */

sendButton.addEventListener("click", async function() {

    if (!selectedFile) {
        return;
    }

    try {

        status.textContent = "Reading JSON...";
        sendButton.disabled = true;


        /* Read the file */

        const text = await selectedFile.text();

        const sensorData = JSON.parse(text);


        /* Basic validation */

        if (!sensorData.data) {

            throw new Error(
                "JSON does not contain a 'data' field."
            );

        }


        status.textContent = "Sending to API...";


        /* Send to FastAPI */

        const response = await fetch(
            API_URL,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(sensorData)
            }
        );


        /* Check HTTP status */

        if (!response.ok) {

            const errorText = await response.text();

            throw new Error(
                `API returned ${response.status}: ${errorText}`
            );

        }


        /* Get API JSON */

        const result = await response.json();


        status.textContent = "Analysis complete.";

        console.log("API Response:", result);


        /* Display result */

        displayResult(result);


    } catch (error) {

        console.error(error);

        status.textContent =
            "Error: " + error.message;

        results.classList.add("hidden");


    } finally {

        sendButton.disabled = false;

    }

});


/* ============================================
   DISPLAY RESULT
============================================ */

function displayResult(data) {

    /*
     * Your API currently returns reasoning as a
     * JSON string.
     *
     * If you change the API to return reasoning
     * as an object, this also supports that.
     */

    let reasoning = data.reasoning;


    /* Parse reasoning if it is a string */

    if (typeof reasoning === "string") {

        try {

            reasoning = JSON.parse(reasoning);

        } catch (error) {

            throw new Error(
                "The API returned invalid reasoning JSON."
            );

        }

    }


    if (!reasoning) {

        throw new Error(
            "API response did not contain reasoning."
        );

    }


    /* ========================================
       DETECTED STATE
    ======================================== */

    document.getElementById(
        "detectedState"
    ).textContent =
        reasoning.detected_state || "--";


    /* ========================================
       CONFIDENCE
    ======================================== */

    document.getElementById(
        "confidence"
    ).textContent =
        reasoning.confidence_percent !== undefined
            ? `${reasoning.confidence_percent}%`
            : "--%";


    /* ========================================
       MODEL AGREEMENT
    ======================================== */

    const agreement =
        reasoning.model_agreement || {};


    document.getElementById(
        "agreementSummary"
    ).textContent =
        agreement.summary || "";


    /* Agreeing models */

    const agreeingList =
        document.getElementById(
            "agreeingModels"
        );

    agreeingList.innerHTML = "";


    (agreement.agreeing_models || [])
        .forEach(model => {

            const li =
                document.createElement("li");

            li.textContent = `✓ ${model}`;

            agreeingList.appendChild(li);

        });


    /* Disagreeing models */

    const disagreeingList =
        document.getElementById(
            "disagreeingModels"
        );

    disagreeingList.innerHTML = "";


    (agreement.disagreeing_models || [])
        .forEach(model => {

            const li =
                document.createElement("li");

            li.textContent = `✗ ${model}`;

            disagreeingList.appendChild(li);

        });


    /* ========================================
       PHYSIOLOGICAL INTERPRETATION
    ======================================== */

    document.getElementById(
        "physiologicalInterpretation"
    ).textContent =
        reasoning.physiological_interpretation || "";


    /* ========================================
       TREND
    ======================================== */

    document.getElementById(
        "trend"
    ).textContent =
        reasoning.trend || "";


    /* ========================================
       USER SUMMARY
    ======================================== */

    document.getElementById(
        "userSummary"
    ).textContent =
        reasoning.user_summary || "";


    /* ========================================
       RAW JSON
    ======================================== */

    document.getElementById(
        "rawJson"
    ).textContent =
        JSON.stringify(data, null, 4);


    /* Show results */

    results.classList.remove("hidden");

}