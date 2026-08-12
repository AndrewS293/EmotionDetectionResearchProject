# ============================================================
# OLLAMA REASONER
# ============================================================
#
# Purpose:
#   Takes the structured output from the ML/ensemble pipeline
#   and uses Ollama to generate a human-readable interpretation.
#
# Pipeline:
#
#   Sensor Data
#       ↓
#   Feature Extraction
#       ↓
#   ML Models
#       ↓
#   Ensemble
#       ↓
#   Prediction Report (JSON/dict)
#       ↓
#   OllamaReasoner
#       ↓
#   LLM Interpretation
#
# IMPORTANT:
#   The LLM is NOT responsible for classification.
#   The ML models perform classification.
#   The LLM interprets the ML results.
#
# ============================================================

import json
from typing import Any, Dict, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "qwen2.5:7b"

DEFAULT_TEMPERATURE = 0.2


# ============================================================
# OLLAMA REASONER
# ============================================================

class OllamaReasoner:

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """
        Initialize the Ollama LLM.

        Parameters
        ----------
        model : str
            Ollama model name.

        temperature : float
            Controls randomness.

            Lower values are preferable here because we want
            consistent interpretations of physiological data.
        """

        self.model_name = model

        self.temperature = temperature

        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
        )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    def _system_prompt(self) -> str:
        """
        Defines the role and behavior of the LLM.
        """

        return """
You are the cognitive reasoning layer of a wearable
physiological emotion detection system.

The system uses machine learning models trained on
physiological sensor data such as:

- Electrodermal Activity (EDA)
- Blood Volume Pulse (BVP)
- Temperature (TEMP)
- Accelerometer (ACC)

The machine learning models are responsible for
classifying the user's physiological state.

You are NOT the classifier.

You must NOT override the machine learning ensemble's
final prediction.

Your responsibility is to interpret the machine learning
results and explain them clearly to the user.

Your reasoning should consider:

1. The final ensemble prediction.
2. The confidence of the ensemble.
3. Individual model predictions.
4. Model confidence values when available.
5. Physiological features.
6. Recent prediction history when available.
7. Whether the state appears sustained or temporary.

When discussing physiological signals:

- EDA can provide information about physiological arousal.
- BVP can provide information related to cardiovascular activity.
- Temperature can provide additional physiological context.
- Accelerometer data can help identify movement/activity.

Do not make medical diagnoses.

Do not claim that a physiological signal proves that
someone is experiencing a specific psychological or medical
condition.

Use language such as:

"the data suggests"

"the detected pattern is consistent with"

"the model indicates"

rather than making absolute medical claims.

If the model confidence is low or the individual models
disagree substantially, explicitly mention that uncertainty.

Keep explanations understandable to a normal user while
still providing enough technical reasoning to be useful.

The final prediction supplied by the ensemble should always
be treated as the official classification.
"""


    # ========================================================
    # FORMAT INPUT
    # ========================================================

    def _format_prediction_report(
        self,
        prediction_report: Dict[str, Any],
    ) -> str:
        """
        Convert the prediction report into readable JSON
        for the LLM.
        """

        return json.dumps(
            prediction_report,
            indent=2,
            default=str,
        )


    # ========================================================
    # BUILD PROMPT
    # ========================================================

    def _build_prompt(
        self,
        prediction_report: Dict[str, Any],
    ) -> str:
        """
        Build the user portion of the LLM prompt.
        """

        formatted_report = self._format_prediction_report(
            prediction_report
        )

        prompt = f"""
Analyze the following output from the physiological
emotion detection system.

The machine learning models have already performed the
classification and the ensemble has produced the final
prediction.

Your task is to explain the result.

Do not perform a new classification.

Do not replace the ensemble prediction.

Instead:

1. State the detected emotional state.
2. Explain the confidence level.
3. Describe which physiological features support the result.
4. Discuss agreement or disagreement between models.
5. Consider recent prediction history if it is provided.
6. Explain whether the result appears sustained or isolated.
7. Provide a short user-friendly interpretation.

Prediction report:

{formatted_report}

Return your response using exactly this structure:

Detected State:
<state>

Confidence:
<confidence description>

Model Agreement:
<brief explanation of model agreement>

Physiological Interpretation:
<explanation of relevant physiological features>

Trend:
<explanation of recent history if available>

User Summary:
<short, natural-language explanation for the user>
"""

        return prompt


    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze_state(
        self,
        prediction_report: Dict[str, Any],
    ) -> str:
        """
        Send a prediction report to Ollama and return the
        generated interpretation.

        Parameters
        ----------
        prediction_report : dict
            Structured output from the ML/ensemble pipeline.

        Returns
        -------
        str
            LLM-generated interpretation.
        """

        if not isinstance(
            prediction_report,
            dict,
        ):
            raise TypeError(
                "prediction_report must be a dictionary."
            )


        system_message = SystemMessage(
            content=self._system_prompt()
        )


        human_message = HumanMessage(
            content=self._build_prompt(
                prediction_report
            )
        )


        response = self.llm.invoke(
            [
                system_message,
                human_message,
            ]
        )


        return response.content


    # ========================================================
    # ANALYZE WITH OPTIONAL USER CONTEXT
    # ========================================================

    def analyze_state_with_context(
        self,
        prediction_report: Dict[str, Any],
        user_context: Optional[str] = None,
    ) -> str:
        """
        Analyze a prediction report while optionally providing
        additional non-sensitive application context.

        Example:

            user_context =
                "The user is viewing their current status."

        This should NOT contain unnecessary personal or medical
        information.
        """

        if user_context:

            prediction_report = dict(
                prediction_report
            )

            prediction_report[
                "application_context"
            ] = user_context


        return self.analyze_state(
            prediction_report
        )


    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def test_connection(self) -> bool:
        """
        Test whether Ollama is responding.

        Returns
        -------
        bool
            True if the model responds successfully.
        """

        try:

            response = self.llm.invoke(
                [
                    HumanMessage(
                        content="Respond with the word READY."
                    )
                ]
            )

            if response and response.content:

                return True

            return False

        except Exception as e:

            print(
                f"Ollama connection failed: {e}"
            )

            return False


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "OLLAMA REASONER TEST"
    )

    print(
        "========================================\n"
    )


    # --------------------------------------------------------
    # Example output from the ML/ensemble pipeline
    # --------------------------------------------------------

    example_report = {

        "final_prediction": "Stress",

        "ensemble_confidence": 0.91,


        "model_predictions": {

            "XGBoost": {

                "prediction": "Stress",

                "confidence": 0.94,

                "weight": 0.35,
            },


            "Random Forest": {

                "prediction": "Stress",

                "confidence": 0.89,

                "weight": 0.25,
            },


            "SVM": {

                "prediction": "Stress",

                "confidence": 0.86,

                "weight": 0.20,
            },


            "KNN": {

                "prediction": "Baseline",

                "confidence": 0.63,

                "weight": 0.10,
            },


            "Logistic Regression": {

                "prediction": "Stress",

                "confidence": 0.82,

                "weight": 0.10,
            },
        },


        "physiological_features": {

            "EDA Mean": 0.82,

            "EDA Std": 0.21,

            "EDA Peak Count": 14,

            "BVP Mean": 88.2,

            "BVP Std": 8.4,

            "TEMP Mean": 31.6,

            "TEMP Std": 0.42,
        },


        "recent_history": {

            "last_5_predictions": [

                "Baseline",

                "Stress",

                "Stress",

                "Stress",

                "Stress",
            ]
        },
    }


    # --------------------------------------------------------
    # Create reasoner
    # --------------------------------------------------------

    reasoner = OllamaReasoner(
        model="qwen2.5:7b",
        temperature=0.2,
    )


    # --------------------------------------------------------
    # Test connection
    # --------------------------------------------------------

    print(
        "Testing Ollama connection..."
    )


    if not reasoner.test_connection():

        print(
            "\nOllama is not responding."
        )

        print(
            "Make sure Ollama is running and the model"
            " has been downloaded."
        )

        raise SystemExit


    print(
        "Ollama connection successful.\n"
    )


    # --------------------------------------------------------
    # Run reasoning
    # --------------------------------------------------------

    print(
        "Sending prediction report to Ollama...\n"
    )


    result = reasoner.analyze_state(
        example_report
    )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print(
        "========================================"
    )

    print(
        "LLM RESPONSE"
    )

    print(
        "========================================\n"
    )

    print(result)

    print(
        "\n========================================"
    )

    print(
        "TEST COMPLETE"
    )

    print(
        "========================================"
    )

#from llm.ollama_reasoner import OllamaReasoner

#reasoner = OllamaReasoner()

#response = reasoner.analyze_state(
#    prediction_report
#)

#print(response)

