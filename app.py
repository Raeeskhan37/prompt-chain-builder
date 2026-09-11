import os
import time
import uuid

import streamlit as st
from groq import Groq


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Prompt Chain Builder",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 0.5rem;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    .app-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }

    .app-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1rem;
    }

    .dashboard {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        background: #f8fafc;
        margin-bottom: 1.2rem;
    }

    .dashboard-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }

    .dashboard-description {
        color: #6b7280;
        font-size: 0.9rem;
    }

    .execution-box {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem;
        background-color: #f8fafc;
        margin-top: 0.8rem;
        margin-bottom: 1rem;
    }

    .step-row {
        padding: 0.55rem 0.75rem;
        border-radius: 9px;
        margin-bottom: 0.3rem;
        font-size: 0.9rem;
    }

    .step-completed {
        background-color: #f0fdf4;
    }

    .step-running {
        background-color: #eff6ff;
    }

    .step-waiting {
        background-color: #f9fafb;
    }

    .step-failed {
        background-color: #fef2f2;
    }

    .step-title {
        font-weight: 600;
    }

    .execution-message {
        margin-top: 0.7rem;
        color: #6b7280;
        font-size: 0.88rem;
    }

    .workflow-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.5rem;
        background-color: #ffffff;
    }

    .workflow-card-title {
        font-weight: 650;
        font-size: 0.98rem;
    }

    .workflow-card-purpose {
        color: #6b7280;
        font-size: 0.84rem;
        margin-top: 0.15rem;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# AI CONFIGURATION
# =========================================================

MODEL_NAME = "openai/gpt-oss-120b"

# Large V3 is preferred here because multilingual
# transcription accuracy is more important than speed.
WHISPER_MODEL = "whisper-large-v3"

MAX_CONTEXT_CHARS = 6000
MAX_OUTPUT_TOKENS = 1200


# =========================================================
# OUTPUT LANGUAGES
# =========================================================

OUTPUT_LANGUAGES = {
    "Auto (same as input)": "Detect the language of the original user request and answer in that same language and script.",
    "English": "English",
    "Urdu": "Urdu using Urdu script (اردو). NEVER use Hindi or Devanagari script.",
    "Pashto": "Pashto using Pashto script (پښتو).",
    "Arabic": "Arabic",
    "Chinese": "Chinese",
    "Spanish": "Spanish",
    "French": "French",
    "German": "German",
    "Turkish": "Turkish",
    "Hindi": "Hindi using Devanagari script.",
    "Bengali": "Bengali",
    "Persian": "Persian",
    "Russian": "Russian",
    "Italian": "Italian",
    "Portuguese": "Portuguese",
    "Japanese": "Japanese",
    "Korean": "Korean",
}


# =========================================================
# GROQ API KEY
# =========================================================

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("Groq API key is not configured.")
    st.info("Please add GROQ_API_KEY in Streamlit Cloud Secrets.")
    st.stop()

client = Groq(api_key=api_key)


# =========================================================
# DEFAULT WORKFLOW STAGES
# =========================================================

DEFAULT_STAGES = [
    {
        "name": "Analysis",
        "purpose": "Understand and analyze the user's request.",
        "instruction": (
            "Analyze the user's request carefully. "
            "Identify the objective, important requirements, "
            "constraints, assumptions, and expected output."
        ),
    },
    {
        "name": "Development",
        "purpose": "Develop the main solution.",
        "instruction": (
            "Using the analysis from the previous stage, "
            "develop a useful and accurate solution. "
            "Include all important details required by the user."
        ),
    },
    {
        "name": "Refinement",
        "purpose": "Improve clarity, completeness and usefulness.",
        "instruction": (
            "Review the previous solution and improve it. "
            "Remove unnecessary content, correct weaknesses, "
            "and make the response clearer and more practical."
        ),
    },
    {
        "name": "Quality Check",
        "purpose": "Check the response for errors and missing requirements.",
        "instruction": (
            "Perform a rigorous quality check of the previous result. "
            "Verify factual claims, dates, days of the week, numbers, names, "
            "calculations, logical consistency, missing requirements and "
            "unsupported assumptions. If an error is found, correct it. "
            "Do not invent information."
        ),
    },
    {
        "name": "Final Answer",
        "purpose": "Produce the final response for the user.",
        "instruction": (
            "Create the final answer using the improved and "
            "quality-checked result. Answer the user's request "
            "directly and clearly."
        ),
    },
]


# =========================================================
# WORKFLOW TEMPLATES
# =========================================================

WORKFLOW_TEMPLATES = {
    "Custom Workflow": {
        "description": "Build a custom multi-stage AI workflow.",
        "stages": [
            {
                "name": "Analysis",
                "purpose": "Understand and analyze the user request.",
                "instruction": (
                    "Analyze the user's request carefully. Identify the "
                    "main objective, requirements, constraints and important "
                    "details needed to solve the task."
                ),
            },
            {
                "name": "Development",
                "purpose": "Develop the main solution.",
                "instruction": (
                    "Develop the main solution based on the user's request "
                    "and the analysis from the previous stage."
                ),
            },
            {
                "name": "Refinement",
                "purpose": "Improve clarity, completeness and usefulness.",
                "instruction": (
                    "Review the solution and improve its clarity, "
                    "completeness, accuracy and usefulness. Remove "
                    "unnecessary material."
                ),
            },
            {
                "name": "Quality Check",
                "purpose": "Check quality and correct problems.",
                "instruction": (
                    "Perform a rigorous quality check. Look for factual "
                    "errors, logical problems, contradictions, missing "
                    "requirements and unclear content. Correct problems "
                    "where possible without inventing information."
                ),
            },
            {
                "name": "Final Answer",
                "purpose": "Produce the final response.",
                "instruction": (
                    "Produce the final answer based on the reviewed solution. "
                    "Follow the selected Output Intent strictly and provide "
                    "only the useful answer to the user."
                ),
            },
        ],
    },

    "Email Writer": {
        "description": "Create clear, professional and well-structured emails.",
        "stages": [
            {
                "name": "Understand Request",
                "purpose": "Understand the email requirements.",
                "instruction": (
                    "Understand the user's purpose, recipient, subject, "
                    "tone, important facts and required action."
                ),
            },
            {
                "name": "Draft Email",
                "purpose": "Create the email draft.",
                "instruction": (
                    "Write a clear and professional email based on the "
                    "user's requirements. Do not invent missing facts."
                ),
            },
            {
                "name": "Review",
                "purpose": "Check the email for quality and accuracy.",
                "instruction": (
                    "Review grammar, clarity, tone, completeness, dates, "
                    "days, numbers, names, times and facts. Check for "
                    "contradictions and unsupported details. Do not invent "
                    "information."
                ),
            },
            {
                "name": "Final Email",
                "purpose": "Produce the polished email.",
                "instruction": (
                    "Produce the final polished email. Follow the selected "
                    "Output Intent strictly. Return only the email content."
                ),
            },
        ],
    },

    "Content Writer": {
        "description": "Create clear, engaging and well-structured content.",
        "stages": [
            {
                "name": "Content Analysis",
                "purpose": "Understand the content requirements.",
                "instruction": (
                    "Analyze the user's topic, audience, purpose, tone and "
                    "required content structure."
                ),
            },
            {
                "name": "Content Creation",
                "purpose": "Create the main content.",
                "instruction": (
                    "Create useful and well-structured content based on the "
                    "user's requirements."
                ),
            },
            {
                "name": "Content Review",
                "purpose": "Improve quality and completeness.",
                "instruction": (
                    "Review the content for accuracy, clarity, "
                    "completeness and engagement. Improve weak areas and "
                    "remove unnecessary material."
                ),
            },
            {
                "name": "Final Content",
                "purpose": "Produce polished final content.",
                "instruction": (
                    "Produce the polished final content with appropriate "
                    "headings, structure and formatting."
                ),
            },
        ],
    },

    "Study Assistant": {
        "description": "Explain topics and create useful learning material.",
        "stages": [
            {
                "name": "Understand Topic",
                "purpose": "Identify learner needs.",
                "instruction": (
                    "Understand the topic, learner level, learning objective "
                    "and important concepts that need to be explained."
                ),
            },
            {
                "name": "Explain",
                "purpose": "Provide a clear explanation.",
                "instruction": (
                    "Explain the topic using simple language. "
                    "Use examples or analogies when they improve understanding."
                ),
            },
            {
                "name": "Learning Review",
                "purpose": "Check understanding and completeness.",
                "instruction": (
                    "Review the explanation for accuracy, clarity and "
                    "completeness. Identify and correct missing important points."
                ),
            },
            {
                "name": "Final Lesson",
                "purpose": "Create useful learning material.",
                "instruction": (
                    "Create learner-friendly learning material using the "
                    "reviewed explanation. Follow the selected Output Intent "
                    "strictly. If Quick Answer is selected, provide only a "
                    "very short explanation with the essential points. Do not "
                    "create a long lesson, cheat-sheet, extended background "
                    "section or unnecessary summary."
                ),
            },
        ],
    },

    "Research Assistant": {
        "description": (
            "Research, analyze, organize and verify information "
            "before producing a final answer."
        ),
        "stages": [
            {
                "name": "Understand Request",
                "purpose": "Understand the research question and requirements.",
                "instruction": (
                    "Understand the user's research question, scope, "
                    "requirements and desired outcome. Identify the key "
                    "points that need to be addressed."
                ),
            },
            {
                "name": "Research & Analyze",
                "purpose": "Analyze the topic and identify important information.",
                "instruction": (
                    "Analyze the research question deeply. Identify relevant "
                    "facts, concepts, relationships, comparisons and evidence "
                    "needed to produce a useful answer. Do not invent facts."
                ),
            },
            {
                "name": "Organize Findings",
                "purpose": "Structure the important findings logically.",
                "instruction": (
                    "Organize the findings into a clear and logical structure. "
                    "Separate key facts, comparisons, evidence and conclusions "
                    "where appropriate."
                ),
            },
            {
                "name": "Fact Check",
                "purpose": "Review accuracy and identify problems.",
                "instruction": (
                    "Critically review the findings for factual errors, "
                    "contradictions, unsupported claims and missing important "
                    "information. Correct errors where the intended correction "
                    "is clear. Do not invent information."
                ),
            },
            {
                "name": "Final Answer",
                "purpose": "Produce the final research answer.",
                "instruction": (
                    "Produce a clear, accurate and well-structured final answer "
                    "based on the reviewed findings. Follow the selected "
                    "Output Intent strictly. Do not add unsupported information."
                ),
            },
        ],
    },
}


# =========================================================
# OUTPUT INTENTS
# =========================================================

OUTPUT_INTENTS = {
    "⚡ Quick Answer": {
        "description": "Short and direct answer with only the essential information.",
        "max_words": 60,
        "instruction": """
Answer very briefly and directly.

STRICT LIMIT: maximum 60 words.

Use only the essential information.
Prefer 1–3 short paragraphs or up to 4 short bullets.
Do not add unnecessary sections, tips, tools, summaries, cheat-sheets,
background information, or extended examples.
Do not repeat the question.
""",
    },

    "🙂 Simple Explanation": {
        "description": "Easy-to-understand explanation for a general user or beginner.",
        "max_words": 220,
        "instruction": (
            "STRICT LENGTH REQUIREMENT: The final answer should normally stay "
            "within 220 words. Explain the answer in simple, clear language "
            "suitable for a beginner. Avoid unnecessary technical terminology. "
            "Use no more than one short example when it improves understanding. "
            "Avoid unnecessary sections or repetition."
        ),
    },

    "📚 Detailed Explanation": {
        "description": "A well-structured explanation with useful details and examples.",
        "max_words": 500,
        "instruction": (
            "Keep the final answer within approximately 500 words. Provide a "
            "well-structured and informative answer. Include important details, "
            "explanations and useful examples while avoiding unnecessary repetition "
            "or complexity."
        ),
    },

    "🔎 Comprehensive Analysis": {
        "description": "Deep and thorough treatment of the user's request.",
        "max_words": 900,
        "instruction": (
            "Keep the final answer within approximately 900 words unless the "
            "user explicitly requires more. Provide a comprehensive and "
            "thorough answer covering relevant details, important considerations, "
            "examples, limitations and supporting explanations."
        ),
    },
}


# =========================================================
# TEMPLATE-SPECIFIC OPTIONS
# =========================================================

EMAIL_STYLES = {
    "Professional": "Professional, clear and respectful.",
    "Formal": "Formal, polished and appropriately respectful.",
    "Friendly": "Warm, friendly and approachable while remaining appropriate.",
    "Concise": "Very concise, direct and professional.",
    "Detailed": "Detailed, complete and professional.",
    "Persuasive": "Professional, respectful and persuasive.",
}

CONTENT_STYLES = {
    "Professional": "Professional and informative.",
    "Engaging": "Engaging, readable and interesting.",
    "Academic": "Academic, structured and informative.",
    "Simple": "Simple, clear and easy to understand.",
}

STUDY_LEVELS = {
    "Beginner": "Explain for a complete beginner.",
    "Intermediate": "Explain for someone with basic knowledge.",
    "Advanced": "Explain with greater technical depth.",
}


# =========================================================
# SESSION STATE
# =========================================================

if "workflow_name" not in st.session_state:
    st.session_state.workflow_name = "My AI Workflow"

if "workflow_description" not in st.session_state:
    st.session_state.workflow_description = "A multi-stage AI workflow."

if "output_intent" not in st.session_state:
    st.session_state.output_intent = "🙂 Simple Explanation"

if "output_language" not in st.session_state:
    st.session_state.output_language = "Auto (same as input)"

if "stages" not in st.session_state:
    st.session_state.stages = [
        {
            "id": str(uuid.uuid4()),
            "name": stage["name"],
            "purpose": stage["purpose"],
            "instruction": stage["instruction"],
        }
        for stage in DEFAULT_STAGES
    ]

if "stage_status" not in st.session_state:
    st.session_state.stage_status = []

if "stage_outputs" not in st.session_state:
    st.session_state.stage_outputs = []

if "final_answer" not in st.session_state:
    st.session_state.final_answer = ""

if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

if "run_completed" not in st.session_state:
    st.session_state.run_completed = False

if "last_template" not in st.session_state:
    st.session_state.last_template = "Custom Workflow"

if "workflow_running" not in st.session_state:
    st.session_state.workflow_running = False

if "template_option" not in st.session_state:
    st.session_state.template_option = "Professional"

if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

if "user_prompt_input" not in st.session_state:
    st.session_state.user_prompt_input = ""


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def limit_context(text, max_chars=MAX_CONTEXT_CHARS):
    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    return text[-max_chars:]


def call_groq(system_prompt, user_prompt, retries=3, max_tokens=None):
    last_error = None

    for attempt in range(retries):

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.4,
                max_tokens=max_tokens or MAX_OUTPUT_TOKENS,
            )

            return response.choices[0].message.content

        except Exception as error:

            last_error = error

            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise last_error


# =========================================================
# VOICE FUNCTIONS
# =========================================================

def transcribe_audio(audio_bytes):
    """
    Automatically detect the spoken language and transcribe
    the speech in the original language/script.

    IMPORTANT:
    This uses the transcription endpoint, NOT the translation
    endpoint. Therefore the speech is not intentionally translated.
    """

    audio_file = (
        "voice_input.wav",
        audio_bytes,
    )

    transcription_prompt = """
Transcribe the speaker's words exactly in the language that was spoken.

IMPORTANT MULTILINGUAL RULES:

- Automatically detect the spoken language.
- Transcribe; do NOT translate.
- Preserve the original language.
- Preserve the original writing system/script.
- If the speaker is speaking Urdu, write the transcription in Urdu script (اردو).
- NEVER convert Urdu speech into Hindi or Devanagari script.
- If the speaker is speaking Pashto, write it in Pashto script (پښتو).
- If the speaker is speaking English, write it in English.
- Do not translate Urdu into Hindi.
- Do not translate Pashto into another language.
- Do not translate English into another language.
- Preserve names, numbers, places and technical terms as spoken.
"""

    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model=WHISPER_MODEL,
        prompt=transcription_prompt,
        response_format="verbose_json",
        temperature=0.0,
    )

    if hasattr(transcription, "text"):
        return transcription.text.strip()

    if isinstance(transcription, str):
        return transcription.strip()

    return str(transcription).strip()


def clean_voice_transcript(transcript):
    """
    Correct obvious speech-to-text mistakes while preserving
    the original language and script.
    """

    if not transcript:
        return ""

    correction_prompt = f"""
You are an intelligent multilingual speech-transcript correction assistant.

The text below was produced by a multilingual speech-to-text model.

Your task is to correct obvious transcription errors while preserving
exactly what the speaker intended.

CRITICAL LANGUAGE RULES:

1. Detect the language of the transcript.
2. Keep the transcript in that SAME language.
3. Keep the SAME writing system/script.
4. NEVER translate the transcript.
5. NEVER convert Urdu into Hindi.
6. If the intended language is Urdu, use Urdu script (اردو), NOT
   Devanagari/Hindi script.
7. If the intended language is Pashto, use Pashto script (پښتو).
8. If the intended language is English, keep it in English.
9. Preserve mixed-language technical terms when they are clearly part
   of the speaker's request.
10. Do not add information.
11. Do not invent names, dates, numbers, places or facts.
12. Correct only obvious speech-recognition, spelling, punctuation or
   grammar mistakes.
13. If the transcript is already correct, leave it essentially unchanged.
14. Return ONLY the corrected transcript.

Speech transcript:
{transcript}
"""

    try:

        corrected = call_groq(
            """You correct multilingual speech-to-text transcripts.
Preserve the original language and script.
Never translate the user's request.
Never convert Urdu into Hindi or Devanagari.""",
            correction_prompt,
            max_tokens=800,
        )

        if corrected:
            return corrected.strip()

    except Exception:
        pass

    return transcript.strip()


def process_voice_input(audio_bytes):
    """
    Full voice pipeline:

    Audio
       ↓
    Automatic language detection
       ↓
    Original-language transcription
       ↓
    Multilingual transcript correction
       ↓
    Same text box
    """

    transcript = transcribe_audio(audio_bytes)

    if not transcript:
        raise ValueError(
            "No speech could be detected in the recording."
        )

    corrected = clean_voice_transcript(
        transcript
    )

    return corrected


def count_words(text):
    if not text:
        return 0

    return len(text.split())


def compress_final_answer(answer, output_intent, template_name):

    intent_config = OUTPUT_INTENTS[output_intent]
    max_words = intent_config["max_words"]

    if not answer:
        return answer

    current_words = count_words(answer)

    if current_words <= max_words:
        return answer.strip()

    compression_prompt = f"""
Rewrite the following final answer into a much shorter final answer.

Workflow:
{template_name}

Output Style:
{output_intent}

ABSOLUTE LIMIT:
Maximum {max_words} words.

IMPORTANT:

- Keep only the information necessary to answer the user's request.
- Do not add new information.
- Do not create sections such as Tips, Tools, Summary, or Cheat-Sheet.
- Remove repetition.
- Remove unnecessary examples.
- Remove background information.
- Use simple language.
- Preserve the SAME OUTPUT LANGUAGE requested by the user.
- If Output Language is Urdu, use Urdu script only.
- NEVER convert Urdu into Hindi or Devanagari.
- For Quick Answer, use 1–3 short paragraphs or up to 4 short bullets.
- The result must be suitable to display directly to the user.

Return ONLY the final answer.

Answer to compress:
{answer}
"""

    try:

        compressed = call_groq(
            """You are a strict final-answer editor.
Make answers concise while preserving meaning.
Always preserve the requested output language and script.
Never translate Urdu into Hindi.""",
            compression_prompt,
        )

        if compressed:
            return compressed.strip()

    except Exception:
        pass

    return answer.strip()


def create_stage(name="New Stage"):
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "purpose": "Define what this stage should accomplish.",
        "instruction": "Explain what the AI should do during this stage.",
    }


def clear_results():
    st.session_state.stage_status = []
    st.session_state.stage_outputs = []
    st.session_state.final_answer = ""
    st.session_state.run_completed = False


def reset_workflow():
    st.session_state.workflow_name = "My AI Workflow"
    st.session_state.workflow_description = "A multi-stage AI workflow."
    st.session_state.output_intent = "🙂 Simple Explanation"
    st.session_state.output_language = "Auto (same as input)"

    st.session_state.stages = [
        {
            "id": str(uuid.uuid4()),
            "name": stage["name"],
            "purpose": stage["purpose"],
            "instruction": stage["instruction"],
        }
        for stage in DEFAULT_STAGES
    ]

    st.session_state.last_template = "Custom Workflow"
    st.session_state.template_option = "Professional"
    st.session_state.workflow_running = False
    st.session_state.voice_transcript = ""
    st.session_state.last_audio_id = None
    st.session_state.user_prompt_input = ""

    clear_results()


def move_stage_up(index):
    if index <= 0:
        return

    stages = st.session_state.stages

    stages[index - 1], stages[index] = (
        stages[index],
        stages[index - 1],
    )

    clear_results()


def move_stage_down(index):
    stages = st.session_state.stages

    if index >= len(stages) - 1:
        return

    stages[index + 1], stages[index] = (
        stages[index],
        stages[index + 1],
    )

    clear_results()


def delete_stage(index):
    if len(st.session_state.stages) <= 2:
        st.warning("A workflow must contain at least 2 stages.")
        return

    del st.session_state.stages[index]
    clear_results()


def add_stage():
    stage_number = len(st.session_state.stages) + 1

    st.session_state.stages.append(
        create_stage(f"Stage {stage_number}")
    )

    clear_results()


def load_template(template_name):
    selected_template = WORKFLOW_TEMPLATES[template_name]

    st.session_state.stages = [
        {
            "id": str(uuid.uuid4()),
            "name": stage["name"],
            "purpose": stage["purpose"],
            "instruction": stage["instruction"],
        }
        for stage in selected_template["stages"]
    ]

    st.session_state.workflow_description = (
        selected_template["description"]
    )

    if template_name == "Custom Workflow":
        st.session_state.workflow_name = "My AI Workflow"
    else:
        st.session_state.workflow_name = template_name

    st.session_state.last_template = template_name
    st.session_state.template_option = "Professional"

    clear_results()


def get_template_option_label(template_name):

    if template_name == "Email Writer":
        return "Email Style"

    if template_name == "Content Writer":
        return "Content Style"

    if template_name == "Study Assistant":
        return "Learning Level"

    return None


def get_template_option_instruction(
    template_name,
    selected_option,
):

    if template_name == "Email Writer":
        return EMAIL_STYLES.get(
            selected_option,
            "",
        )

    if template_name == "Content Writer":
        return CONTENT_STYLES.get(
            selected_option,
            "",
        )

    if template_name == "Study Assistant":
        return STUDY_LEVELS.get(
            selected_option,
            "",
        )

    return ""


def render_execution_tracker(status_placeholder):

    total = len(st.session_state.stages)

    completed = sum(
        1
        for status in st.session_state.stage_status
        if status == "completed"
    )

    current_stage = None

    for index, status in enumerate(
        st.session_state.stage_status
    ):

        if status == "running":
            current_stage = index
            break

    progress_value = (
        completed / total
        if total
        else 0
    )

    with status_placeholder.container():

        st.markdown(
            '<div class="execution-box">',
            unsafe_allow_html=True,
        )

        st.markdown("### 🔄 Workflow Execution")

        st.progress(progress_value)

        for index, stage in enumerate(
            st.session_state.stages
        ):

            status = (
                st.session_state.stage_status[index]
                if index < len(st.session_state.stage_status)
                else "pending"
            )

            if status == "completed":
                icon = "✅"
                state_text = "Completed"
                css_class = "step-completed"

            elif status == "running":
                icon = "🔄"
                state_text = "Running"
                css_class = "step-running"

            elif status == "failed":
                icon = "❌"
                state_text = "Failed"
                css_class = "step-failed"

            else:
                icon = "⏳"
                state_text = "Waiting"
                css_class = "step-waiting"

            st.markdown(
                f"""
                <div class="step-row {css_class}">
                    {icon}
                    <span class="step-title">
                        Step {index + 1}/{total}
                    </span>
                    — {stage['name']}
                    <span style="float:right;">
                        {state_text}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if current_stage is not None:

            st.markdown(
                f"""
                <div class="execution-message">
                    Processing:
                    <strong>
                        {st.session_state.stages[current_stage]['name']}
                    </strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🔗 AI Prompt Chain Builder")

    st.markdown("---")

    st.markdown("### Workflow")

    st.write(
        "Create a workflow by connecting multiple AI stages. "
        "Each stage receives the output of the previous stage."
    )

    st.markdown("### How It Works")

    st.markdown(
    """
    **1. Choose Workflow**  
    Select what you want ChainForge to do.

    **2. Provide Information**  
    Type or speak your request naturally.

    **3. Run Workflow**  
    ChainForge processes your request through multiple AI stages automatically.

    **4. Get Final Answer**  
    The final stage produces the completed result.
    """
)

    st.markdown("---")

    st.markdown("### 🤖 AI Configuration")

    st.caption(f"Model: `{MODEL_NAME}`")

    st.caption(
        f"Voice model: `{WHISPER_MODEL}`"
    )

    st.caption(
        "Voice language: Automatic detection"
    )

    st.caption(
        f"Maximum output tokens: `{MAX_OUTPUT_TOKENS}`"
    )

    st.markdown("---")

    if st.button(
        "🔄 Reset Workflow",
        use_container_width=True,
    ):

        reset_workflow()
        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="app-title">🔗 AI Prompt Chain Builder</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "Build, customize and execute multi-stage AI workflows."
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# WORKFLOW TEMPLATE
# =========================================================

template_options = list(
    WORKFLOW_TEMPLATES.keys()
)

template_name = st.selectbox(
    "🧰 Workflow",
    template_options,
    index=(
        template_options.index(
            st.session_state.last_template
        )
        if st.session_state.last_template
        in template_options
        else 0
    ),
    disabled=st.session_state.workflow_running,
)

if (
    template_name
    != st.session_state.last_template
    and not st.session_state.workflow_running
):

    load_template(template_name)
    st.rerun()


# =========================================================
# DASHBOARD
# =========================================================

stages = st.session_state.stages

completed_count = sum(
    1
    for status in st.session_state.stage_status
    if status == "completed"
)

if st.session_state.run_completed:
    workflow_status = "Completed"

elif st.session_state.workflow_running:
    workflow_status = "Running"

elif any(
    status == "failed"
    for status in st.session_state.stage_status
):
    workflow_status = "Failed"

else:
    workflow_status = "Ready"


st.markdown(
    f"""
    <div class="dashboard">
        <div class="dashboard-title">
            {st.session_state.workflow_name}
        </div>
        <div class="dashboard-description">
            {st.session_state.workflow_description}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Stages",
        len(stages),
    )

with metric2:
    st.metric(
        "Completed",
        completed_count,
    )

with metric3:
    st.metric(
        "Status",
        workflow_status,
    )


# =========================================================
# TEMPLATE-SPECIFIC USER OPTIONS
# =========================================================

option_label = get_template_option_label(
    template_name
)

if option_label:

    if template_name == "Email Writer":
        option_values = list(
            EMAIL_STYLES.keys()
        )

    elif template_name == "Content Writer":
        option_values = list(
            CONTENT_STYLES.keys()
        )

    else:
        option_values = list(
            STUDY_LEVELS.keys()
        )

    selected_option = st.selectbox(
        option_label,
        option_values,
        index=(
            option_values.index(
                st.session_state.template_option
            )
            if st.session_state.template_option
            in option_values
            else 0
        ),
        disabled=st.session_state.workflow_running,
    )

    if (
        selected_option
        != st.session_state.template_option
        and not st.session_state.workflow_running
    ):

        st.session_state.template_option = (
            selected_option
        )

        clear_results()


# =========================================================
# OUTPUT LANGUAGE
# =========================================================

output_language = st.selectbox(
    "🌐 Output Language",
    list(OUTPUT_LANGUAGES.keys()),
    index=(
        list(OUTPUT_LANGUAGES.keys()).index(
            st.session_state.output_language
        )
        if st.session_state.output_language
        in OUTPUT_LANGUAGES
        else 0
    ),
    disabled=st.session_state.workflow_running,
)

if (
    output_language
    != st.session_state.output_language
    and not st.session_state.workflow_running
):

    st.session_state.output_language = output_language
    clear_results()

if output_language == "Auto (same as input)":

    st.caption(
        "The AI will detect the language of your request and "
        "answer in the same language."
    )

else:

    st.caption(
        f"Final answers will be produced in {output_language}."
    )


# =========================================================
# OUTPUT INTENT
# =========================================================

output_intent = st.selectbox(
    "🎯 Answer Style",
    list(OUTPUT_INTENTS.keys()),
    index=list(
        OUTPUT_INTENTS.keys()
    ).index(
        st.session_state.output_intent
    ),
    disabled=st.session_state.workflow_running,
)

if (
    output_intent
    != st.session_state.output_intent
    and not st.session_state.workflow_running
):

    st.session_state.output_intent = (
        output_intent
    )

    clear_results()

st.caption(
    OUTPUT_INTENTS[output_intent]["description"]
)


# =========================================================
# CUSTOM WORKFLOW INFORMATION
# =========================================================

if template_name == "Custom Workflow":

    with st.expander(
        "⚙️ Custom Workflow Information",
        expanded=True,
    ):

        col1, col2 = st.columns(2)

        with col1:

            workflow_name = st.text_input(
                "Workflow Name",
                value=st.session_state.workflow_name,
                disabled=st.session_state.workflow_running,
            )

            if (
                workflow_name
                != st.session_state.workflow_name
                and not st.session_state.workflow_running
            ):

                st.session_state.workflow_name = (
                    workflow_name
                )

                clear_results()

        with col2:

            workflow_description = st.text_input(
                "Workflow Description",
                value=(
                    st.session_state.workflow_description
                ),
                disabled=st.session_state.workflow_running,
            )

            if (
                workflow_description
                != st.session_state.workflow_description
                and not st.session_state.workflow_running
            ):

                st.session_state.workflow_description = (
                    workflow_description
                )

                clear_results()


# =========================================================
# VOICE INPUT
# =========================================================

st.markdown("## 🎤 Voice Input")

st.caption(
    "Speak naturally in any supported language. "
    "The language is detected automatically and your speech "
    "is transcribed in the same language and script."
)

audio_value = st.audio_input(
    "🎙️ Speak your request",
    disabled=st.session_state.workflow_running,
)


# =========================================================
# PROCESS NEW AUDIO
# =========================================================

if (
    audio_value is not None
    and not st.session_state.workflow_running
):

    audio_bytes = audio_value.getvalue()

    audio_id = hash(audio_bytes)

    if (
        audio_id
        != st.session_state.last_audio_id
    ):

        with st.spinner(
            "🎤 Detecting language and converting your voice to text..."
        ):

            try:

                corrected_text = process_voice_input(
                    audio_bytes
                )

                if not corrected_text:
                    raise ValueError(
                        "No usable text was returned from the recording."
                    )

                # IMPORTANT:
                # This happens BEFORE the text_area widget
                # is created below.
                st.session_state.voice_transcript = (
                    corrected_text
                )

                st.session_state.last_prompt = (
                    corrected_text
                )

                st.session_state.user_prompt_input = (
                    corrected_text
                )

                st.session_state.last_audio_id = (
                    audio_id
                )

                clear_results()

                st.success(
                    "✅ Voice converted automatically. "
                    "You can edit the text below if needed."
                )

            except Exception as error:

                st.session_state.last_audio_id = (
                    audio_id
                )

                st.error(
                    f"❌ Voice processing failed: {error}"
                )


# =========================================================
# USER REQUEST
# =========================================================

st.markdown(
    "## 📝 What would you like to process?"
)

placeholder_text = (
    "Example: Write an email requesting leave from my manager "
    "for three days next week."
    if template_name == "Email Writer"
    else
    "Example: Explain the primary purpose of the human brain."
    if template_name == "Study Assistant"
    else
    "Example: Write an article about the impact of artificial "
    "intelligence on education."
    if template_name == "Content Writer"
    else
    "Example: Research the benefits and risks of artificial intelligence."
    if template_name == "Research Assistant"
    else
    "Enter or speak the request you want your workflow to process."
)

user_prompt = st.text_area(
    "Describe your request",
    height=150,
    placeholder=placeholder_text,
    key="user_prompt_input",
    label_visibility="collapsed",
    disabled=st.session_state.workflow_running,
)


# =========================================================
# EXECUTION AREA
# =========================================================

execution_container = st.empty()


# =========================================================
# GENERATE BUTTON
# =========================================================

if not st.session_state.workflow_running:

    if template_name == "Email Writer":
        button_text = "✨ Generate Email"

    elif template_name == "Content Writer":
        button_text = "✨ Generate Content"

    elif template_name == "Study Assistant":
        button_text = "✨ Generate Lesson"

    elif template_name == "Research Assistant":
        button_text = "🔎 Run Research Workflow"

    elif template_name == "Custom Workflow":
        button_text = "▶️ Run Workflow"

    else:
        button_text = "✨ Generate"

    run_clicked = st.button(
        button_text,
        type="primary",
        use_container_width=True,
    )

else:

    run_clicked = False


# =========================================================
# START WORKFLOW
# =========================================================

if run_clicked:

    if not user_prompt.strip():

        st.warning(
            "Please enter or record your request before generating a result."
        )

    elif len(st.session_state.stages) < 2:

        st.warning(
            "A workflow must contain at least 2 stages."
        )

    else:

        st.session_state.last_prompt = (
            user_prompt
        )

        total_stages = len(
            st.session_state.stages
        )

        st.session_state.stage_status = [
            "pending"
            for _ in range(total_stages)
        ]

        st.session_state.stage_outputs = [
            ""
            for _ in range(total_stages)
        ]

        st.session_state.final_answer = ""

        st.session_state.run_completed = False

        st.session_state.workflow_running = True

        status_placeholder = (
            execution_container.empty()
        )

        render_execution_tracker(
            status_placeholder
        )

        output_config = OUTPUT_INTENTS[
            st.session_state.output_intent
        ]

        output_instruction = (
            output_config["instruction"]
        )

        template_option_instruction = (
            get_template_option_instruction(
                template_name,
                st.session_state.template_option,
            )
        )

        # -------------------------------------------------
        # OUTPUT LANGUAGE INSTRUCTION
        # -------------------------------------------------

        selected_output_language = (
            st.session_state.output_language
        )

        if (
            selected_output_language
            == "Auto (same as input)"
        ):

            language_instruction = """
OUTPUT LANGUAGE:

Automatically identify the language and script of the
Original User Request.

The final answer MUST use the same language and script
as the user's original request.

Examples:

- Urdu input → Urdu script.
- Pashto input → Pashto script.
- English input → English.
- Arabic input → Arabic.
- Roman Urdu input → Roman Urdu unless the user explicitly
  requests another language.

CRITICAL:

Urdu is NOT Hindi.

If the user speaks or writes Urdu, respond in Urdu script
(اردو), NOT Hindi and NOT Devanagari.

Do not translate the user's language unless the user explicitly
requests translation.
"""

        else:

            requested_language_description = (
                OUTPUT_LANGUAGES[
                    selected_output_language
                ]
            )

            language_instruction = f"""
OUTPUT LANGUAGE:

The user explicitly selected:

{selected_output_language}

The final answer MUST be written in:
{requested_language_description}

This output-language instruction has priority over the
detected input language.

Do NOT answer in the input language if it differs from
the selected output language.

CRITICAL:

If the selected output language is Urdu, use Urdu script
(اردو) only.

NEVER convert Urdu into Hindi or Devanagari.

Return the final user-facing answer ONLY in the selected
output language.
"""

        workflow_failed = False


        # =================================================
        # EXECUTE STAGES SEQUENTIALLY
        # =================================================

        for index, stage in enumerate(
            st.session_state.stages
        ):

            stage_number = index + 1

            st.session_state.stage_status[
                index
            ] = "running"

            render_execution_tracker(
                status_placeholder
            )

            previous_output = ""

            if index > 0:

                previous_output = (
                    st.session_state.stage_outputs[
                        index - 1
                    ]
                )

                previous_output = limit_context(
                    previous_output
                )


            # ---------------------------------------------
            # CURRENT STAGE INPUT
            # ---------------------------------------------

            stage_input = f"""
Original User Request:
{user_prompt}

Selected Workflow:
{template_name}

Selected Workflow Option:
{st.session_state.template_option}

Selected Output Intent:
{st.session_state.output_intent}

Selected Output Language:
{selected_output_language}

LANGUAGE REQUIREMENT:
{language_instruction}

Previous Stage Output:
{
    previous_output
    if previous_output
    else
    "This is the first stage. There is no previous stage output."
}

Current Stage:
{stage['name']}

Current Stage Purpose:
{stage['purpose']}

Now perform the current stage according to its instructions
and the selected workflow options.
"""


            # ---------------------------------------------
            # FINAL STAGE CONSTRAINT
            # ---------------------------------------------

            final_stage_instruction = ""

            if index == total_stages - 1:

                max_words = output_config[
                    "max_words"
                ]

                final_stage_instruction = f"""
FINAL OUTPUT REQUIREMENTS:

The response produced by this stage will be shown directly
to the user.

The selected Answer Style is:
{st.session_state.output_intent}

The selected Output Language is:
{selected_output_language}

STRICT MAXIMUM LENGTH:
The final answer must not exceed {max_words} words.

Do not ignore this limit even if earlier stages contain much
more information.

Use only the information necessary to satisfy the user's request.

Do not copy the full previous stage output.

Do not add unnecessary background information.

Do not add repeated summaries.

Do not add a long introduction.

Do not create unnecessary sections.

LANGUAGE:

{language_instruction}

Return ONLY the final user-facing answer.
"""


            # ---------------------------------------------
            # SYSTEM PROMPT
            # ---------------------------------------------

            system_prompt = f"""
You are Stage {stage_number} of a multi-stage AI workflow.

Workflow Name:
{st.session_state.workflow_name}

Workflow Description:
{st.session_state.workflow_description}

Selected Workflow:
{template_name}

Selected Workflow Option:
{st.session_state.template_option}

Workflow Option Instructions:
{template_option_instruction}

Current Stage Name:
{stage['name']}

Current Stage Purpose:
{stage['purpose']}

Current Stage Instructions:
{stage['instruction']}

Selected Output Intent:
{st.session_state.output_intent}

Output Intent Instructions:
{output_instruction}

Selected Output Language:
{selected_output_language}

{language_instruction}

{final_stage_instruction}

Your role is to complete ONLY the current stage effectively.

The output of this stage will be passed to the next stage.

The selected workflow option controls the style, tone or
level appropriate for this workflow.

The selected Output Intent controls the desired depth,
complexity, length and presentation style of the final answer.

Follow the stage instructions carefully.

Do not ignore important information from the user's request.

Do not invent facts or personal details that are not supported
by the user's request or the previous stage output.

When checking factual information, pay special attention to
dates, days of the week, numbers, names, calculations,
logical consistency and unsupported assumptions.

LANGUAGE SAFETY:

- Never randomly change the user's language.
- Never translate unless requested or unless the selected
  Output Language explicitly requires it.
- Urdu must remain Urdu when Urdu is selected.
- Urdu must use Urdu script.
- Urdu must NEVER become Hindi or Devanagari.
- Pashto must remain Pashto when Pashto is selected.
- Preserve the selected output language through every stage.
- The Final Answer stage has the highest priority for output language.

Produce useful output that the next stage can directly use.
"""


            # ---------------------------------------------
            # CALL AI
            # ---------------------------------------------

            try:

                output = call_groq(
                    system_prompt,
                    stage_input,
                )

                if not output:

                    raise ValueError(
                        "The AI returned an empty response."
                    )


                # -----------------------------------------
                # ENFORCE FINAL ANSWER LENGTH
                # -----------------------------------------

                if index == total_stages - 1:

                    output = compress_final_answer(
                        output,
                        st.session_state.output_intent,
                        template_name,
                    )


                st.session_state.stage_outputs[
                    index
                ] = output

                st.session_state.stage_status[
                    index
                ] = "completed"

                render_execution_tracker(
                    status_placeholder
                )

            except Exception as error:

                st.session_state.stage_status[
                    index
                ] = "failed"

                workflow_failed = True

                render_execution_tracker(
                    status_placeholder
                )

                status_placeholder.error(
                    f"❌ Step {stage_number}/{total_stages} "
                    f"failed: {error}"
                )

                break


        # =================================================
        # WORKFLOW COMPLETION
        # =================================================

        if not workflow_failed:

            st.session_state.final_answer = (
                st.session_state.stage_outputs[-1]
            )

            st.session_state.run_completed = True

            st.session_state.workflow_running = False

            render_execution_tracker(
                status_placeholder
            )

            status_placeholder.success(
                f"🎉 All {total_stages} stages completed successfully."
            )

        else:

            st.session_state.workflow_running = False

            status_placeholder.error(
                "Workflow stopped because a stage failed."
            )

        time.sleep(0.5)

        st.rerun()


# =========================================================
# FINAL ANSWER
# =========================================================

if st.session_state.final_answer:

    st.markdown("---")

    st.markdown(
        "## 🎯 Final Answer"
    )

    answer_words = count_words(
        st.session_state.final_answer
    )

    with st.container(border=True):

        st.success(
            "✅ Workflow Completed Successfully"
        )

        st.caption(
            f"{st.session_state.workflow_name} • "
            f"{len(st.session_state.stages)} stages • "
            f"{st.session_state.output_intent} • "
            f"Output: {st.session_state.output_language} • "
            f"{answer_words} words"
        )

        st.markdown(
            st.session_state.final_answer
        )

    st.markdown("")

    download_col, regenerate_col = (
        st.columns(2)
    )

    with download_col:

        st.download_button(
            "📥 Download Answer",
            data=st.session_state.final_answer,
            file_name="chainforge_final_answer.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with regenerate_col:

        if st.button(
            "🔄 Regenerate",
            use_container_width=True,
            key="final_answer_regenerate",
        ):

            st.session_state.run_completed = False
            st.session_state.final_answer = ""
            st.session_state.stage_status = []
            st.session_state.stage_outputs = []

            st.rerun()


# =========================================================
# INTERNAL STAGE RESULTS
# =========================================================

if (
    st.session_state.stage_outputs
    and any(
        st.session_state.stage_outputs
    )
):

    with st.expander(
        "📊 View Internal Stage Results",
        expanded=False,
    ):

        st.caption(
            "Intermediate results are normally hidden "
            "to keep the interface clean."
        )

        for index, output in enumerate(
            st.session_state.stage_outputs
        ):

            if not output:
                continue

            stage_name = (
                st.session_state.stages[index]["name"]
                if index
                < len(
                    st.session_state.stages
                )
                else f"Stage {index + 1}"
            )

            with st.expander(
                f"Step {index + 1}: {stage_name}",
                expanded=False,
            ):

                st.write(output)


# =========================================================
# WORKFLOW SUMMARY
# =========================================================

if st.session_state.run_completed:

    with st.expander(
        "📋 Workflow Summary",
        expanded=False,
    ):

        summary_col1, summary_col2 = (
            st.columns(2)
        )

        with summary_col1:

            st.write(
                f"**Workflow:** "
                f"{st.session_state.workflow_name}"
            )

            st.write(
                f"**Stages:** "
                f"{len(st.session_state.stages)}"
            )

            st.write(
                "**Status:** Completed"
            )

        with summary_col2:

            st.write(
                f"**Model:** {MODEL_NAME}"
            )

            st.write(
                f"**Output Language:** "
                f"{st.session_state.output_language}"
            )

            st.write(
                f"**Output Intent:** "
                f"{st.session_state.output_intent}"
            )

            st.write(
                "**Execution:** Sequential"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        AI Prompt Chain Builder • Multi-stage AI Workflow Engine
    </div>
    """,
    unsafe_allow_html=True,
)
