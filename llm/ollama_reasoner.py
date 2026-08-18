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

STRICT OUTPUT RULES:

The machine learning ensemble is the authoritative
source for classification.

You are NOT allowed to change or reinterpret the
ensemble's final prediction.

You are NOT allowed to calculate a new confidence.

You are NOT allowed to introduce emotion classes
that do not appear in the prediction report.

You are NOT allowed to invent physiological
measurements or trends.

Your role is explanation only.

Python code outside of the LLM is responsible for
the final prediction, confidence, and model agreement.

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
        Build a constrained prompt for the LLM.
    
        The LLM is ONLY responsible for explaining the
        prediction. It does not determine the prediction
        or confidence.
        """
    
        formatted_report = self._format_prediction_report(
            prediction_report
        )
    
        prompt = f"""
    You are the explanation layer of a physiological
    emotion detection system.
    
    The machine learning ensemble is the authoritative
    classifier.
    
    Your job is ONLY to explain the results provided below.
    
    IMPORTANT RULES:
    
    1. Do NOT change the final prediction.
    2. Do NOT create a new emotion class.
    3. Do NOT calculate or modify the ensemble confidence.
    4. Do NOT invent model predictions.
    5. Do NOT invent confidence values.
    6. Do NOT invent physiological measurements.
    7. Do NOT claim that a physiological feature changed
       unless that information is explicitly present in the
       prediction report.
    8. Do NOT make medical diagnoses.
    9. If insufficient physiological information is provided,
       explicitly say so.
    10. Treat the values in the prediction report as factual
        outputs from the machine learning system.
    
    The following prediction report is authoritative:
    
    {formatted_report}
    
    Generate ONLY an explanation of the results.
    
    Return valid JSON with EXACTLY these fields:
    
    {{
        "model_agreement_summary": "<brief explanation of how the models agreed or disagreed>",
        "physiological_interpretation": "<brief explanation based ONLY on physiological information explicitly provided>",
        "trend": "<brief explanation of recent prediction history if provided>",
        "user_summary": "<short user-friendly explanation>"
    }}
    
    Do not include:
    
    - detected_state
    - confidence_percent
    - alternative emotion labels
    - invented sensor values
    - invented model predictions
    
    Do not include markdown.
    
    Do not include ```json.
    
    Return ONLY the JSON object.
    """

    

        return prompt


    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze_state(
        self,
        prediction_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze a prediction report using Ollama.
    
        The ML ensemble remains authoritative for the
        prediction and confidence. Ollama only generates
        explanatory information.
        """
    
        if not isinstance(
            prediction_report,
            dict,
        ):
            raise TypeError(
                "prediction_report must be a dictionary."
            )
    
        # --------------------------------------------------------
        # AUTHORITATIVE ML OUTPUTS
        # --------------------------------------------------------

        # Your prediction report may contain the ensemble information
        # either at the top level or inside an "ensemble" dictionary.

        ensemble = prediction_report.get("ensemble", {})

        if not isinstance(ensemble, dict):
            ensemble = {}


        # --------------------------------------------------------
        # FINAL PREDICTION
        # --------------------------------------------------------

        final_prediction = (
            ensemble.get("prediction")
            or prediction_report.get("prediction")
            or prediction_report.get("final_prediction")
        )


        # --------------------------------------------------------
        # ENSEMBLE CONFIDENCE
        # --------------------------------------------------------

        ensemble_confidence = (
            ensemble.get("confidence")
            if ensemble.get("confidence") is not None
            else prediction_report.get("confidence")
        )

        if ensemble_confidence is None:
            ensemble_confidence = prediction_report.get(
                "ensemble_confidence"
            )


        # --------------------------------------------------------
        # MODEL PREDICTIONS
        # --------------------------------------------------------

        models = prediction_report.get(
            "models",
            {}
        )

        if not isinstance(models, dict):
            models = {}


        model_predictions = models.get(
            "predictions",
            {}
        )


        # Support the older report format too
        if not model_predictions:

            model_predictions = prediction_report.get(
                "model_predictions",
                {}
            )


        # --------------------------------------------------------
        # SAFETY CHECK
        # --------------------------------------------------------

        if final_prediction is None:

            raise ValueError(
                "Could not find ensemble prediction in prediction report. "
                f"Available keys: {list(prediction_report.keys())}"
            )


        if ensemble_confidence is None:

            raise ValueError(
                "Could not find ensemble confidence in prediction report. "
                f"Available keys: {list(prediction_report.keys())}"
            )


        # --------------------------------------------------------
        # CALCULATE MODEL AGREEMENT IN PYTHON
        # --------------------------------------------------------

        agreeing_models = [
            name
            for name, prediction
            in model_predictions.items()
            if prediction == final_prediction
        ]

        disagreeing_models = [
            name
            for name, prediction
            in model_predictions.items()
            if prediction != final_prediction
        ]


        model_agreement = {

            "summary": (
                f"{len(agreeing_models)} of "
                f"{len(model_predictions)} models "
                f"predicted {final_prediction}."
            ),

            "agreeing_models": agreeing_models,

            "disagreeing_models": disagreeing_models,
        }
                # --------------------------------------------------------
        # SEND REPORT TO OLLAMA
        # --------------------------------------------------------
    
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
    
        # --------------------------------------------------------
        # PARSE LLM RESPONSE
        # --------------------------------------------------------
    
        try:
    
            llm_result = json.loads(
                response.content
            )
    
        except json.JSONDecodeError:
    
            # Don't allow malformed LLM output to break
            # the authoritative ML result.
    
            llm_result = {
                "model_agreement_summary": "",
                "physiological_interpretation": "",
                "trend": "",
                "user_summary": response.content,
            }
    
        # --------------------------------------------------------
        # BUILD FINAL RESPONSE
        # --------------------------------------------------------
    
        final_response = {
    
            # THESE COME DIRECTLY FROM THE ML PIPELINE
            "detected_state": final_prediction,
    
            "confidence_percent": round(
                ensemble_confidence * 100
            ),
    
            # THIS IS CALCULATED BY PYTHON
            "model_agreement": {
    
                "summary": model_agreement[
                    "summary"
                ],
    
                "agreeing_models": agreeing_models,
    
                "disagreeing_models": disagreeing_models,
            },
    
            # THESE COME FROM OLLAMA
            "physiological_interpretation": (
                llm_result.get(
                    "physiological_interpretation",
                    ""
                )
            ),
    
            "trend": (
                llm_result.get(
                    "trend",
                    ""
                )
            ),
    
            "user_summary": (
                llm_result.get(
                    "user_summary",
                    ""
                )
            ),
        }
    
        return final_response

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




    
#from llm.ollama_reasoner import OllamaReasoner

#reasoner = OllamaReasoner()

#response = reasoner.analyze_state(
#    prediction_report
#)

#print(response)