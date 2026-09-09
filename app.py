import os
import time
import uuid

import streamlit as st
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Prompt Chain Builder",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

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


# ============================================================
# GROQ CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"

MAX_CONTEXT_CHARS = 6000
MAX_OUTPUT_TOKENS = 1200

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("Groq API key is not configured.")
    st.info(
        "Please add GROQ_API_KEY in Streamlit Cloud Secrets."
    )
    st.stop()

client = Groq(api_key=api_key)


# ============================================================
# DEFAULT STAGES
# ============================================================

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
            "Perform a quality check of the previous result. "
            "Look for factual problems, missing requirements, "
            "logical issues, unclear wording, and inconsistencies."
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
# -----------------------------
# Workflow Templates
# -----------------------------

WORKFLOW_TEMPLATES = {
    "Custom Workflow": {
        "description": "Build your own multi-stage AI workflow.",
        "stages": [
            {
                "name": "Analysis",
                "purpose": "Understand and analyze the user's request.",
                "instruction": "Analyze the user's request carefully. Identify the objective, important requirements, constraints, assumptions, and expected output.",
            },
            {
                "name": "Development",
                "purpose": "Develop the main solution.",
                "instruction": "Using the analysis from the previous stage, develop a useful and accurate solution. Include all important details required by the user.",
            },
            {
                "name": "Refinement",
                "purpose": "Improve clarity, completeness and usefulness.",
                "instruction": "Review the previous solution and improve it. Remove unnecessary content, correct weaknesses, and make the response clearer and more practical.",
            },
            {
                "name": "Quality Check",
                "purpose": "Check the response for errors and missing requirements.",
                "instruction": "Perform a quality check of the previous result. Look for factual problems, missing requirements, logical issues, unclear wording, and inconsistencies.",
            },
            {
                "name": "Final Answer",
                "purpose": "Produce the final response for the user.",
                "instruction": "Create the final answer using the improved and quality-checked result. Answer the user's request directly and clearly.",
            },
        ],
    },

    "Email Writer": {
        "description": "Create professional emails quickly.",
        "stages": [
            {
                "name": "Understand Request",
                "purpose": "Understand the purpose, recipient and required tone.",
                "instruction": "Identify the purpose of the email, recipient, important facts, desired tone and required action.",
            },
            {
                "name": "Draft Email",
                "purpose": "Create the main email.",
                "instruction": "Write a professional, clear and concise email using the information provided.",
            },
            {
                "name": "Review",
                "purpose": "Improve professionalism and clarity.",
                "instruction": "Review the email for grammar, clarity, tone, completeness and professionalism. Improve it where necessary.",
            },
            {
                "name": "Final Email",
                "purpose": "Produce the final ready-to-send email.",
                "instruction": "Return the polished final email ready for the user to copy and send.",
            },
        ],
    },

    "Content Writer": {
        "description": "Create well-structured articles and content.",
        "stages": [
            {
                "name": "Content Analysis",
                "purpose": "Understand the topic and audience.",
                "instruction": "Analyze the requested topic, target audience, purpose, tone and important points that should be covered.",
            },
            {
                "name": "Content Creation",
                "purpose": "Write the main content.",
                "instruction": "Create useful, engaging and well-structured content based on the analysis.",
            },
            {
                "name": "Content Review",
                "purpose": "Improve quality and readability.",
                "instruction": "Review the content for accuracy, structure, readability, completeness and engagement. Improve weak areas.",
            },
            {
                "name": "Final Content",
                "purpose": "Produce polished final content.",
                "instruction": "Produce the final polished content with appropriate headings, structure and formatting.",
            },
        ],
    },

    "Study Assistant": {
        "description": "Explain topics and create useful learning material.",
        "stages": [
            {
                "name": "Understand Topic",
                "purpose": "Identify the learner's needs.",
                "instruction": "Understand the topic, learner level, learning objective and important concepts that need to be explained.",
            },
            {
                "name": "Explain",
                "purpose": "Create a clear explanation.",
                "instruction": "Explain the topic in simple language appropriate for the learner. Use examples and analogies where helpful.",
            },
            {
                "name": "Learning Review",
                "purpose": "Check understanding and completeness.",
                "instruction": "Review the explanation for accuracy, clarity and completeness. Identify anything important that is missing.",
            },
            {
                "name": "Final Lesson",
                "purpose": "Produce useful learning material.",
                "instruction": "Create the final learner-friendly explanation with examples, key points and a concise summary.",
            },
        ],
    },
}

# ============================================================
# SESSION STATE
# ============================================================

if "workflow_name" not in st.session_state:
    st.session_state.workflow_name = "My AI Workflow"

if "workflow_description" not in st.session_state:
    st.session_state.workflow_description = (
        "A multi-stage AI workflow."
    )

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def limit_context(text, max_chars=MAX_CONTEXT_CHARS):
    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    return text[-max_chars:]


def call_groq(system_prompt, user_prompt, retries=3):
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
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "purpose": "Define what this stage should accomplish.",
        "instruction": (
            "Explain what the AI should do during this stage."
        ),
    }


def clear_results():
    st.session_state.stage_status = []
    st.session_state.stage_outputs = []
    st.session_state.final_answer = ""
    st.session_state.run_completed = False


def reset_workflow():
    st.session_state.workflow_name = "My AI Workflow"

    st.session_state.workflow_description = (
        "A multi-stage AI workflow."
    )

    st.session_state.stages = [
        {
            "id": str(uuid.uuid4()),
            "name": stage["name"],
            "purpose": stage["purpose"],
            "instruction": stage["instruction"],
        }
        for stage in DEFAULT_STAGES
    ]

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
        st.warning(
            "A workflow must contain at least 2 stages."
        )
        return

    del st.session_state.stages[index]

    clear_results()


def add_stage():
    st.session_state.stages.append(
        create_stage(
            f"Stage {len(st.session_state.stages) + 1}"
        )
    )

    clear_results()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🔗 AI Prompt Chain Builder")

    st.markdown("---")

    st.markdown("### Workflow")

    st.write(
        "Create and execute multi-stage AI workflows."
    )

    st.markdown("---")

    st.markdown("### How It Works")

    st.write(
        "1. Define your workflow"
    )

    st.write(
        "2. Configure the stages"
    )

    st.write(
        "3. Enter your request"
    )

    st.write(
        "4. Run the workflow"
    )

    st.write(
        "5. Review the final answer"
    )

    st.markdown("---")

    st.markdown("### AI Configuration")

    st.caption(
        f"Model: {MODEL_NAME}"
    )

    st.caption(
        f"Maximum output tokens: {MAX_OUTPUT_TOKENS}"
    )

    st.markdown("---")

    if st.button(
        "🔄 Reset Workflow",
        use_container_width=True,
    ):
        reset_workflow()
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "# 🔗 AI Prompt Chain Builder"
)

st.markdown(
    "Build, customize and execute multi-stage AI workflows."
)

st.divider()


# ============================================================
# WORKFLOW INFORMATION
# ============================================================

st.markdown("## ⚙️ Workflow Information")

col1, col2 = st.columns(2)

with col1:
    workflow_name = st.text_input(
        "Workflow Name",
        value=st.session_state.workflow_name,
        key="workflow_name_input",
    )

    if workflow_name != st.session_state.workflow_name:
        st.session_state.workflow_name = workflow_name
        clear_results()


with col2:
    workflow_description = st.text_input(
        "Workflow Description",
        value=st.session_state.workflow_description,
        key="workflow_description_input",
    )

    if (
        workflow_description
        != st.session_state.workflow_description
    ):
        st.session_state.workflow_description = (
            workflow_description
        )
        clear_results()


# ============================================================
# WORKFLOW METRICS
# ============================================================

st.markdown("## 📊 Workflow Overview")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Stages",
        len(st.session_state.stages),
    )

with metric2:
    completed_count = sum(
        1
        for status in st.session_state.stage_status
        if status == "completed"
    )

    st.metric(
        "Completed",
        completed_count,
    )

with metric3:
    if st.session_state.run_completed:
        status_text = "Completed"
    elif any(
        status == "running"
        for status in st.session_state.stage_status
    ):
        status_text = "Running"
    elif any(
        status == "failed"
        for status in st.session_state.stage_status
    ):
        status_text = "Failed"
    else:
        status_text = "Ready"

    st.metric(
        "Status",
        status_text,
    )

# -----------------------------
# Workflow Template
# -----------------------------

st.markdown("## 🧰 Workflow Template")

template_name = st.selectbox(
    "Choose a workflow template",
    list(WORKFLOW_TEMPLATES.keys()),
)

st.caption(
    WORKFLOW_TEMPLATES[template_name]["description"]
)
if "last_template" not in st.session_state:
    st.session_state.last_template = template_name

if template_name != st.session_state.last_template:

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

st.rerun()
# ============================================================
# WORKFLOW BUILDER
# ============================================================
st.markdown("## 🧩 Workflow Builder")

st.caption(
    "Configure the stages that will process the user's request."
)


for index, stage in enumerate(
    st.session_state.stages
):

    stage_number = index + 1

    # --------------------------------------------------------
    # STAGE HEADER
    # --------------------------------------------------------

    status = "pending"

    if (
        index < len(st.session_state.stage_status)
    ):
        status = st.session_state.stage_status[index]

    if status == "completed":
        status_icon = "✅"
    elif status == "running":
        status_icon = "🔄"
    elif status == "failed":
        status_icon = "❌"
    else:
        status_icon = "⚪"

    st.markdown(
        f"### {status_icon} Stage {stage_number} — {stage['name']}"
    )

    st.caption(stage["purpose"])

    # --------------------------------------------------------
    # STAGE CONTROLS
    # --------------------------------------------------------

    control1, control2, control3, control4 = st.columns(
        [1, 1, 1, 1]
    )

    with control1:
        if st.button(
            "⬆️ Move Up",
            key=f"move_up_{stage['id']}",
            disabled=index == 0,
            use_container_width=True,
        ):
            move_stage_up(index)
            st.rerun()

    with control2:
        if st.button(
            "⬇️ Move Down",
            key=f"move_down_{stage['id']}",
            disabled=index
            == len(st.session_state.stages) - 1,
            use_container_width=True,
        ):
            move_stage_down(index)
            st.rerun()

    with control3:
        if st.button(
            "🗑️ Delete",
            key=f"delete_{stage['id']}",
            disabled=len(st.session_state.stages) <= 2,
            use_container_width=True,
        ):
            delete_stage(index)
            st.rerun()

    with control4:
        st.write("")

    # --------------------------------------------------------
    # STAGE EDITOR
    # --------------------------------------------------------

    with st.expander(
        f"✏️ Edit Stage {stage_number}: {stage['name']}",
        expanded=index == 0,
    ):

        name_key = f"stage_name_{stage['id']}"
        purpose_key = f"stage_purpose_{stage['id']}"
        instruction_key = (
            f"stage_instruction_{stage['id']}"
        )

        new_name = st.text_input(
            "Stage Name",
            value=stage["name"],
            key=name_key,
        )

        new_purpose = st.text_input(
            "Stage Purpose",
            value=stage["purpose"],
            key=purpose_key,
        )

        new_instruction = st.text_area(
            "Stage Instructions",
            value=stage["instruction"],
            height=140,
            key=instruction_key,
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

    # --------------------------------------------------------
    # STAGE OUTPUT
    # --------------------------------------------------------

    if index < len(
        st.session_state.stage_outputs
    ):

        output = st.session_state.stage_outputs[index]

        with st.expander(
            f"📄 Stage {stage_number} Output",
            expanded=False,
        ):

            st.write(output)

    st.divider()


# ============================================================
# ADD STAGE
# ============================================================

if len(st.session_state.stages) < 5:

    if st.button(
        "➕ Add Stage",
        use_container_width=True,
    ):
        add_stage()
        st.rerun()

else:

    st.info(
        "Maximum of 5 stages reached."
    )


# ============================================================
# USER REQUEST
# ============================================================

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


# ============================================================
# RUN WORKFLOW
# ============================================================

st.markdown("## ▶️ Execute Workflow")

run_button = st.button(
    "🚀 Run Workflow",
    type="primary",
    use_container_width=True,
)


if run_button:

    if not user_prompt.strip():

        st.warning(
            "Please enter a request before running the workflow."
        )

    elif len(st.session_state.stages) < 2:

        st.warning(
            "A workflow must contain at least 2 stages."
        )

    else:

        st.session_state.last_prompt = user_prompt

        st.session_state.stage_status = [
            "pending"
            for _ in st.session_state.stages
        ]

        st.session_state.stage_outputs = [
            ""
            for _ in st.session_state.stages
        ]

        st.session_state.final_answer = ""

        st.session_state.run_completed = False

        current_context = user_prompt

        progress_bar = st.progress(
            0,
            text="Starting workflow..."
        )

        status_placeholder = st.empty()

        total_stages = len(
            st.session_state.stages
        )

        workflow_failed = False

        for index, stage in enumerate(
            st.session_state.stages
        ):

            stage_number = index + 1

            st.session_state.stage_status[index] = (
                "running"
            )

            status_placeholder.info(
                f"Running Stage {stage_number}: "
                f"{stage['name']}"
            )

            system_prompt = f"""
You are executing Stage {stage_number}
of a multi-stage AI workflow.

Stage Name:
{stage['name']}

Stage Purpose:
{stage['purpose']}

Stage Instructions:
{stage['instruction']}

You must follow the stage instructions carefully.

The output of this stage will be passed to
the next stage of the workflow.
"""

            if index == 0:

                stage_input = f"""
User Request:

{user_prompt}
"""

            else:

                previous_output = (
                    st.session_state.stage_outputs[
                        index - 1
                    ]
                )

                previous_output = limit_context(
                    previous_output
                )

                stage_input = f"""
Original User Request:

{user_prompt}

Previous Stage Output:

{previous_output}

Now perform the current stage.
"""

            try:

                output = call_groq(
                    system_prompt,
                    stage_input,
                )

                st.session_state.stage_outputs[
                    index
                ] = output

                st.session_state.stage_status[
                    index
                ] = "completed"

                current_context = output

                progress = (
                    stage_number / total_stages
                )

                progress_bar.progress(
                    progress,
                    text=(
                        f"Completed Stage "
                        f"{stage_number} of "
                        f"{total_stages}"
                    ),
                )

            except Exception as error:

                st.session_state.stage_status[
                    index
                ] = "failed"

                workflow_failed = True

                st.error(
                    f"Stage {stage_number} failed: "
                    f"{error}"
                )

                break

        if not workflow_failed:

            st.session_state.final_answer = (
                st.session_state.stage_outputs[-1]
            )

            st.session_state.run_completed = True

            progress_bar.progress(
                1.0,
                text="Workflow completed successfully."
            )

            status_placeholder.success(
                "✅ Workflow completed successfully."
            )
        else:

            progress_bar.empty()

            status_placeholder.error(
                "❌ Workflow stopped because a stage failed."
            )

        # Refresh the dashboard so Workflow Overview
        # shows the updated execution status.
        st.rerun()


# ============================================================
# FINAL ANSWER
# ============================================================

if st.session_state.final_answer:

    st.markdown("## 🎯 Final Answer")

    st.success(
        "Workflow completed successfully."
    )

    st.write(
        st.session_state.final_answer
    )

    st.download_button(
        "⬇️ Download Final Answer",
        data=st.session_state.final_answer,
        file_name="workflow_result.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================
# STAGE RESULTS
# ============================================================

if (
    st.session_state.stage_outputs
    and any(st.session_state.stage_outputs)
):

    st.markdown("## 📚 Stage Results")

    for index, output in enumerate(
        st.session_state.stage_outputs
    ):

        if not output:
            continue

        stage_name = (
            st.session_state.stages[index]["name"]
        )

        with st.expander(
            f"Stage {index + 1}: {stage_name}"
        ):

            st.write(output)


# ============================================================
# WORKFLOW SUMMARY
# ============================================================

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

    with summary_col2:

        st.write(
            "**Status:** Completed"
        )

        st.write(
            f"**Model:** {MODEL_NAME}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Prompt Chain Builder • "
    "Multi-stage AI workflow orchestration"
)
