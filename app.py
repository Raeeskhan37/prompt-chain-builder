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
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .app-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .metric-box {
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background-color: #f8fafc;
        text-align: center;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .stage-status {
        padding: 0.35rem 0.7rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
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
MAX_CONTEXT_CHARS = 6000
MAX_OUTPUT_TOKENS = 1200


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
            "unsupported assumptions. Pay special attention to information "
            "that could be objectively verified. If an error is found, "
            "provide the corrected information and explain what must be changed. "
            "Do not merely say that the result looks good. "
            "Return a clear corrected version or precise corrections that the "
            "Final Answer stage can directly use."
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
        "description": "Build your own multi-stage AI workflow.",
        "stages": [
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
                    "unsupported assumptions. Pay special attention to information "
                    "that could be objectively verified. If an error is found, "
                    "provide the corrected information and explain what must be changed. "
                    "Do not merely say that the result looks good. "
                    "Return a clear corrected version or precise corrections that the "
                    "Final Answer stage can directly use."
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
        ],
    },

    "Email Writer": {
        "description": "Create professional emails quickly.",
        "stages": [
            {
                "name": "Understand Request",
                "purpose": "Understand the purpose, recipient and required tone.",
                "instruction": (
                    "Understand the user's email request. Identify the purpose, "
                    "recipient, important facts, requested action and appropriate tone."
                ),
            },
            {
                "name": "Draft Email",
                "purpose": "Create the main email.",
                "instruction": (
                    "Create a professional, clear and concise email using "
                    "the information provided by the user."
                ),
            },
            {
                "name": "Review",
                "purpose": "Verify the email for accuracy, clarity and professionalism.",
                "instruction": (
                    "Perform a rigorous review of the email. Check grammar, "
                    "clarity, tone, completeness and professionalism. Verify "
                    "dates, days of the week, numbers, names, times and factual "
                    "details. Identify and correct contradictions, incorrect "
                    "assumptions or unsupported details. Do not invent information "
                    "that the user did not provide. If information is missing, "
                    "use a suitable placeholder or neutral wording. Return the "
                    "corrected email for the Final Email stage."
                ),
            },
            {
                "name": "Final Email",
                "purpose": "Produce the accurate, polished and ready-to-send email.",
                "instruction": (
                    "Use the reviewed result to produce the final polished email. "
                    "Apply all corrections and do not reintroduce errors. Preserve "
                    "the user's facts and do not invent missing personal details. "
                    "Use placeholders such as [Boss's Name] or [Your Name] when "
                    "necessary. Return only the final polished email."
                ),
            },
        ],
    },

    "Content Writer": {
        "description": "Create well-structured articles and content.",
        "stages": [
            {
                "name": "Content Analysis",
                "purpose": "Understand the topic and audience.",
                "instruction": (
                    "Analyze the requested topic, target audience, purpose, "
                    "tone and important points that should be covered."
                ),
            },
            {
                "name": "Content Creation",
                "purpose": "Write the main content.",
                "instruction": (
                    "Create useful, engaging and well-structured content "
                    "based on the analysis and user's request."
                ),
            },
            {
                "name": "Content Review",
                "purpose": "Improve quality and readability.",
                "instruction": (
                    "Review the content for accuracy, structure, readability, "
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
                    "reviewed explanation. Include examples, key points "
                    "and a concise summary when appropriate."
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
        "description": (
            "Short and direct answer with only the essential information."
        ),
        "instruction": (
            "Keep the final answer concise. Give only the essential "
            "information needed to answer the user's request. Avoid "
            "unnecessary background, long explanations and excessive examples."
        ),
    },

    "🙂 Simple Explanation": {
        "description": (
            "Easy-to-understand explanation for a general user or beginner."
        ),
        "instruction": (
            "Explain the answer in simple, clear language suitable for a "
            "beginner. Avoid unnecessary technical terminology. Use a short "
            "example when it improves understanding."
        ),
    },

    "📚 Detailed Explanation": {
        "description": (
            "A well-structured explanation with useful details and examples."
        ),
        "instruction": (
            "Provide a well-structured and informative answer. Include the "
            "important details, explanations and examples needed for good "
            "understanding, while avoiding unnecessary complexity."
        ),
    },

    "🔎 Comprehensive Analysis": {
        "description": (
            "Deep and thorough treatment of the user's request."
        ),
        "instruction": (
            "Provide a comprehensive and thorough answer. Cover relevant "
            "details, important considerations, examples, limitations and "
            "supporting explanations. Do not omit important information."
        ),
    },
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


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def limit_context(text, max_chars=MAX_CONTEXT_CHARS):
    """Limit the amount of previous-stage text passed to the next stage."""
    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    return text[-max_chars:]


def call_groq(system_prompt, user_prompt, retries=3):
    """Call Groq with simple retry handling."""

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
                max_tokens=MAX_OUTPUT_TOKENS,
            )

            return response.choices[0].message.content

        except Exception as error:
            last_error = error

            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise last_error


def create_stage(name="New Stage"):
    """Create a new workflow stage."""

    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "purpose": "Define what this stage should accomplish.",
        "instruction": "Explain what the AI should do during this stage.",
    }


def clear_results():
    """Clear previous execution results."""

    st.session_state.stage_status = []
    st.session_state.stage_outputs = []
    st.session_state.final_answer = ""
    st.session_state.run_completed = False


def reset_workflow():
    """Restore the default workflow."""

    st.session_state.workflow_name = "My AI Workflow"
    st.session_state.workflow_description = "A multi-stage AI workflow."
    st.session_state.output_intent = "🙂 Simple Explanation"

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

    clear_results()


def move_stage_up(index):
    """Move a stage one position upward."""

    if index <= 0:
        return

    stages = st.session_state.stages

    stages[index - 1], stages[index] = (
        stages[index],
        stages[index - 1],
    )

    clear_results()


def move_stage_down(index):
    """Move a stage one position downward."""

    stages = st.session_state.stages

    if index >= len(stages) - 1:
        return

    stages[index + 1], stages[index] = (
        stages[index],
        stages[index + 1],
    )

    clear_results()


def delete_stage(index):
    """Delete a stage while keeping at least two stages."""

    if len(st.session_state.stages) <= 2:
        st.warning("A workflow must contain at least 2 stages.")
        return

    del st.session_state.stages[index]

    clear_results()


def add_stage():
    """Add a new stage."""

    stage_number = len(st.session_state.stages) + 1

    st.session_state.stages.append(
        create_stage(f"Stage {stage_number}")
    )

    clear_results()


def load_template(template_name):
    """Load a selected workflow template."""

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

    clear_results()


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
        **1. Choose Template**  
        Select a workflow type.

        **2. Configure Stages**  
        Customize what each AI stage does.

        **3. Provide Information**  
        Enter the request you want to process.

        **4. Run Workflow**  
        AI processes the request stage by stage.

        **5. Get Final Answer**  
        The final stage produces the completed result.
        """
    )

    st.markdown("---")

    st.markdown("### 🤖 AI Configuration")

    st.caption(f"Model: `{MODEL_NAME}`")
    st.caption(f"Maximum output tokens: `{MAX_OUTPUT_TOKENS}`")

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

st.markdown("# 🔗 AI Prompt Chain Builder")

st.markdown(
    "Build, customize and execute multi-stage AI workflows."
)

st.divider()


# =========================================================
# WORKFLOW TEMPLATE
# =========================================================

st.markdown("## 🧰 Workflow Template")

template_options = list(WORKFLOW_TEMPLATES.keys())

template_name = st.selectbox(
    "Choose a workflow template",
    template_options,
    index=(
        template_options.index(st.session_state.last_template)
        if st.session_state.last_template in template_options
        else 0
    ),
)

st.caption(
    WORKFLOW_TEMPLATES[template_name]["description"]
)

if template_name != st.session_state.last_template:

    load_template(template_name)

    st.rerun()


# =========================================================
# OUTPUT INTENT
# =========================================================

st.markdown("## 🎯 Output Intent")

output_intent = st.selectbox(
    "How should the workflow answer?",
    list(OUTPUT_INTENTS.keys()),
    index=list(OUTPUT_INTENTS.keys()).index(
        st.session_state.output_intent
    ),
)

if output_intent != st.session_state.output_intent:

    st.session_state.output_intent = output_intent

    clear_results()

st.caption(
    OUTPUT_INTENTS[output_intent]["description"]
)


# =========================================================
# WORKFLOW INFORMATION
# =========================================================

st.markdown("## ⚙️ Workflow Information")

col1, col2 = st.columns(2)

with col1:

    workflow_name = st.text_input(
        "Workflow Name",
        value=st.session_state.workflow_name,
    )

    if workflow_name != st.session_state.workflow_name:

        st.session_state.workflow_name = workflow_name

        clear_results()


with col2:

    workflow_description = st.text_input(
        "Workflow Description",
        value=st.session_state.workflow_description,
    )

    if workflow_description != st.session_state.workflow_description:

        st.session_state.workflow_description = workflow_description

        clear_results()


# =========================================================
# WORKFLOW METRICS
# =========================================================

stages = st.session_state.stages

completed_count = sum(
    1
    for status in st.session_state.stage_status
    if status == "completed"
)

if st.session_state.run_completed:

    workflow_status = "Completed"

elif any(
    status == "running"
    for status in st.session_state.stage_status
):

    workflow_status = "Running"

elif any(
    status == "failed"
    for status in st.session_state.stage_status
):

    workflow_status = "Failed"

else:

    workflow_status = "Ready"


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
# WORKFLOW BUILDER
# =========================================================

st.markdown("## 🧩 Workflow Builder")

for index, stage in enumerate(st.session_state.stages):

    stage_number = index + 1

    if index < len(st.session_state.stage_status):

        current_status = st.session_state.stage_status[index]

    else:

        current_status = "pending"


    if current_status == "completed":

        status_icon = "✅"

    elif current_status == "running":

        status_icon = "🔄"

    elif current_status == "failed":

        status_icon = "❌"

    else:

        status_icon = "⏳"


    st.markdown(
        f"### {status_icon} Stage {stage_number}: {stage['name']}"
    )

    st.caption(
        f"Purpose: {stage['purpose']}"
    )

    control1, control2, control3 = st.columns(
        [1, 1, 1]
    )

    with control1:

        if st.button(
            "⬆️ Move Up",
            key=f"up_{stage['id']}",
            disabled=(index == 0),
            use_container_width=True,
        ):

            move_stage_up(index)

            st.rerun()


    with control2:

        if st.button(
            "⬇️ Move Down",
            key=f"down_{stage['id']}",
            disabled=(index == len(stages) - 1),
            use_container_width=True,
        ):

            move_stage_down(index)

            st.rerun()


    with control3:

        if st.button(
            "🗑️ Delete",
            key=f"delete_{stage['id']}",
            use_container_width=True,
        ):

            delete_stage(index)

            st.rerun()


    with st.expander(
        f"✏️ Edit Stage {stage_number}",
        expanded=False,
    ):

        new_name = st.text_input(
            "Stage Name",
            value=stage["name"],
            key=f"name_{stage['id']}",
        )

        new_purpose = st.text_area(
            "Stage Purpose",
            value=stage["purpose"],
            key=f"purpose_{stage['id']}",
            height=90,
        )

        new_instruction = st.text_area(
            "AI Instruction",
            value=stage["instruction"],
            key=f"instruction_{stage['id']}",
            height=160,
        )

        if (
            new_name != stage["name"]
            or new_purpose != stage["purpose"]
            or new_instruction != stage["instruction"]
        ):

            stage["name"] = new_name
            stage["purpose"] = new_purpose
            stage["instruction"] = new_instruction

            clear_results()

    if (
        index < len(st.session_state.stage_outputs)
        and st.session_state.stage_outputs[index]
    ):

        with st.expander(
            f"📄 Stage {stage_number} Output",
            expanded=False,
        ):

            st.write(
                st.session_state.stage_outputs[index]
            )

    st.divider()


# =========================================================
# ADD STAGE
# =========================================================

if len(st.session_state.stages) < 5:

    if st.button(
        "➕ Add Stage",
        use_container_width=True,
    ):

        add_stage()

        st.rerun()

else:

    st.info(
        "Maximum of 5 stages allowed in the current workflow."
    )


# =========================================================
# USER REQUEST
# =========================================================

st.markdown("## 📝 User Request")

user_prompt = st.text_area(
    "What would you like the workflow to process?",
    value=st.session_state.last_prompt,
    height=180,
    placeholder=(
        "Example: Create a professional report about "
        "the impact of artificial intelligence on education."
    ),
    key="user_prompt_input",
)


# =========================================================
# RUN WORKFLOW
# =========================================================

st.markdown("## ▶️ Execute Workflow")

if st.button(
    "🚀 Run Workflow",
    type="primary",
    use_container_width=True,
):

    if not user_prompt.strip():

        st.warning(
            "Please enter a user request before running the workflow."
        )

    elif len(st.session_state.stages) < 2:

        st.warning(
            "A workflow must contain at least 2 stages."
        )

    else:

        # -------------------------------------------------
        # INITIALIZE EXECUTION STATE
        # -------------------------------------------------

        st.session_state.last_prompt = user_prompt

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

        progress_bar = st.progress(
            0,
            text="Starting workflow...",
        )

        status_placeholder = st.empty()

        workflow_failed = False

        output_instruction = OUTPUT_INTENTS[
            st.session_state.output_intent
        ]["instruction"]


        # -------------------------------------------------
        # EXECUTE EACH STAGE
        # -------------------------------------------------

        for index, stage in enumerate(
            st.session_state.stages
        ):

            stage_number = index + 1

            st.session_state.stage_status[index] = "running"

            status_placeholder.info(
                f"Running Stage {stage_number} of "
                f"{total_stages}: {stage['name']}"
            )


            # -------------------------------------------------
            # STEP 6: PASS PREVIOUS STAGE OUTPUT
            # -------------------------------------------------

            previous_output = ""

            if index > 0:

                previous_output = (
                    st.session_state.stage_outputs[index - 1]
                )

                previous_output = limit_context(
                    previous_output
                )


            stage_input = f"""
Original User Request:
{user_prompt}

Selected Output Intent:
{st.session_state.output_intent}

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
and the selected Output Intent.
"""


            # -------------------------------------------------
            # SYSTEM PROMPT
            # -------------------------------------------------

            system_prompt = f"""
You are Stage {stage_number} of a multi-stage AI workflow.

Workflow Name:
{st.session_state.workflow_name}

Workflow Description:
{st.session_state.workflow_description}

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

Your role is to complete ONLY the current stage effectively.

The output of this stage will be passed to the next stage.

The selected Output Intent controls the desired depth,
complexity and presentation style of the eventual answer.

Follow the stage instructions carefully.

Do not ignore important information from the user's request.

Do not invent facts or personal details that are not supported
by the user's request or the previous stage output.

When checking factual information, pay special attention to
dates, days of the week, numbers, names, calculations,
logical consistency and unsupported assumptions.

Produce useful output that the next stage can directly use.
"""


            # -------------------------------------------------
            # CALL AI
            # -------------------------------------------------

            try:

                output = call_groq(
                    system_prompt,
                    stage_input,
                )

                if not output:
                    raise ValueError(
                        "The AI returned an empty response."
                    )


                st.session_state.stage_outputs[index] = output

                st.session_state.stage_status[index] = (
                    "completed"
                )


                progress = (
                    stage_number / total_stages
                )

                progress_bar.progress(
                    progress,
                    text=(
                        f"Completed Stage {stage_number} "
                        f"of {total_stages}"
                    ),
                )


            except Exception as error:

                st.session_state.stage_status[index] = (
                    "failed"
                )

                workflow_failed = True

                status_placeholder.error(
                    f"Stage {stage_number} failed: {error}"
                )

                break


        # -------------------------------------------------
        # WORKFLOW COMPLETION
        # -------------------------------------------------

        if not workflow_failed:

            st.session_state.final_answer = (
                st.session_state.stage_outputs[-1]
            )

            st.session_state.run_completed = True

            progress_bar.progress(
                1.0,
                text="Workflow completed successfully.",
            )

            status_placeholder.success(
                "All workflow stages completed successfully."
            )

        else:

            status_placeholder.error(
                "Workflow stopped because a stage failed."
            )


        # -------------------------------------------------
        # REFRESH PAGE
        # -------------------------------------------------

        time.sleep(0.5)

        st.rerun()


# =========================================================
# FINAL ANSWER
# =========================================================

if st.session_state.final_answer:

    st.markdown("## 🎯 Final Answer")

    st.success(
        "Workflow completed successfully."
    )

    st.write(
        st.session_state.final_answer
    )

    st.download_button(
        label="⬇️ Download Result as TXT",
        data=st.session_state.final_answer,
        file_name="workflow_result.txt",
        mime="text/plain",
        use_container_width=True,
    )


# =========================================================
# STAGE RESULTS
# =========================================================

if (
    st.session_state.stage_outputs
    and any(st.session_state.stage_outputs)
):

    st.markdown("## 📊 Stage Results")

    for index, output in enumerate(
        st.session_state.stage_outputs
    ):

        if not output:
            continue

        if index < len(st.session_state.stages):

            stage_name = (
                st.session_state.stages[index]["name"]
            )

        else:

            stage_name = f"Stage {index + 1}"


        with st.expander(
            f"Stage {index + 1}: {stage_name}",
            expanded=False,
        ):

            st.write(output)


# =========================================================
# WORKFLOW SUMMARY
# =========================================================

if st.session_state.run_completed:

    st.markdown("## 📋 Workflow Summary")

    summary_col1, summary_col2 = st.columns(2)

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
