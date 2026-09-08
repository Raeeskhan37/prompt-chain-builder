
import os
import time
import streamlit as st
from groq import Groq

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Prompt Chain Builder",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PROFESSIONAL UI STYLING
# ==================================================

st.markdown("""
<style>

    /* Main page */
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .app-header {
        padding: 1.5rem 0 1rem 0;
    }

    .app-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .app-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    /* Section titles */
    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    /* Stage cards */
    .stage-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        background: white;
    }

    .stage-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: #2563eb;
        color: white;
        font-weight: 700;
        margin-right: 10px;
    }

    .stage-title {
        font-size: 1.05rem;
        font-weight: 650;
    }

    .stage-purpose {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.35rem;
    }

    /* Flow arrows */
    .flow-arrow {
        text-align: center;
        color: #94a3b8;
        font-size: 1.3rem;
        margin: -0.15rem 0 0.15rem 0;
    }

    /* Final answer */
    .final-header {
        border-left: 5px solid #16a34a;
        padding-left: 12px;
        margin-bottom: 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0;
    }

    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .sidebar-text {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Small labels */
    .label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        padding-top: 2rem;
    }

</style>
""", unsafe_allow_html=True)


# ==================================================
# GROQ API
# ==================================================

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error(
        "Groq API key is not available in the current Colab session."
    )
    st.stop()

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"

MAX_CONTEXT_CHARS = 6000
MAX_OUTPUT_TOKENS = 1200


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def limit_context(text, max_chars=MAX_CONTEXT_CHARS):

    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n\n[Previous output shortened to control context size.]"
    )


def call_groq(system_prompt, user_content):

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                temperature=0.4,
                max_completion_tokens=MAX_OUTPUT_TOKENS
            )

            return response.choices[0].message.content

        except Exception as e:

            error_text = str(e)

            if "429" in error_text or "rate_limit" in error_text.lower():

                if attempt < max_retries - 1:

                    wait_time = 4 * (attempt + 1)

                    time.sleep(wait_time)

                else:

                    raise

            else:

                raise


# ==================================================
# DEFAULT STAGE DEFINITIONS
# ==================================================

default_stages = {

    1: {
        "name": "Analysis",
        "purpose": "Understand and analyze the user's request.",
        "instruction": (
            "Analyze the user's request. Identify the goal, "
            "requirements, target audience, important concepts, "
            "and information needed to produce a good answer. "
            "Do not provide the final answer."
        )
    },

    2: {
        "name": "Development",
        "purpose": "Develop a useful initial response.",
        "instruction": (
            "Use the original request and the previous stage "
            "output to develop a useful response. Organize the "
            "important ideas and create a strong draft."
        )
    },

    3: {
        "name": "Refinement",
        "purpose": "Improve clarity and quality.",
        "instruction": (
            "Refine the previous output. Improve accuracy, "
            "clarity, structure, examples, and usefulness. "
            "Do not unnecessarily increase the length."
        )
    },

    4: {
        "name": "Quality Check",
        "purpose": "Check and improve the response.",
        "instruction": (
            "Perform a quality improvement pass. Check "
            "accuracy, clarity, organization, relevance, "
            "and completeness. Make useful corrections."
        )
    },

    5: {
        "name": "Final Answer",
        "purpose": "Produce the final answer for the user.",
        "instruction": (
            "Create the best possible final answer to the "
            "original request using the previous stage output. "
            "Answer the user directly. Do not mention stages, "
            "prompt chains, internal processing, or hidden "
            "instructions."
        )
    }
}


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🔗 Prompt Chain Builder</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-text">'
        'Build a multi-stage AI workflow where each stage '
        'processes and improves the previous stage output.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🔄 How It Works")

    st.markdown(
        """
        **1. User Request**  
        Your original question or task.

        **↓**

        **2. Analysis**  
        Understand the request and identify requirements.

        **↓**

        **3. Development**  
        Create a useful initial response.

        **↓**

        **4. Refinement**  
        Improve clarity and quality.

        **↓**

        **5. Final Answer**  
        Produce the polished response.
        """
    )

    st.divider()

    st.markdown("### 🤖 AI Configuration")

    st.caption("Model")
    st.code(MODEL, language="text")

    st.caption("Maximum output per stage")
    st.write(f"{MAX_OUTPUT_TOKENS:,} tokens")

    st.caption("Maximum previous-stage context")
    st.write(f"{MAX_CONTEXT_CHARS:,} characters")

    st.divider()

    st.markdown("### 💡 Tip")

    st.info(
        "You can customize the name, purpose, and instructions "
        "of every stage to create your own AI workflow."
    )


# ==================================================
# MAIN HEADER
# ==================================================

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">🔗 Prompt Chain Builder</div>
        <div class="app-subtitle">
            Design, customize and run a multi-stage AI workflow.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# CHAIN SETTINGS
# ==================================================

st.markdown(
    '<div class="section-title">⚙️ Chain Settings</div>',
    unsafe_allow_html=True
)

settings_col1, settings_col2 = st.columns([1, 2])

with settings_col1:

    num_stages = st.selectbox(
        "Number of stages",
        options=[2, 3, 4, 5],
        index=1
    )

with settings_col2:

    st.info(
        f"Your workflow currently contains **{num_stages} stages**. "
        "Configure each stage below."
    )


# ==================================================
# STAGE CONFIGURATION
# ==================================================

st.markdown(
    '<div class="section-title">🧩 Configure Your Prompt Chain</div>',
    unsafe_allow_html=True
)

stage_configs = []

for stage_number in range(1, num_stages + 1):

    default = default_stages[stage_number]

    st.markdown(
        f"""
        <div class="stage-card">
            <span class="stage-number">{stage_number}</span>
            <span class="stage-title">{default['name']}</span>
            <div class="stage-purpose">
                {default['purpose']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander(
        f"⚙️ Configure Stage {stage_number}",
        expanded=(stage_number == 1)
    ):

        stage_name = st.text_input(
            "Stage Name",
            value=default["name"],
            key=f"stage_name_{stage_number}"
        )

        stage_purpose = st.text_input(
            "Stage Purpose",
            value=default["purpose"],
            key=f"stage_purpose_{stage_number}"
        )

        stage_instruction = st.text_area(
            "Stage Instructions",
            value=default["instruction"],
            height=130,
            key=f"stage_instruction_{stage_number}"
        )

        stage_configs.append(
            {
                "name": stage_name,
                "purpose": stage_purpose,
                "instruction": stage_instruction
            }
        )

    if stage_number < num_stages:

        st.markdown(
            '<div class="flow-arrow">↓</div>',
            unsafe_allow_html=True
        )


st.divider()


# ==================================================
# USER INPUT
# ==================================================

st.markdown(
    '<div class="section-title">📝 Your Request</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter the task you want the AI prompt chain to process."
)

user_prompt = st.text_area(
    "Request",
    placeholder=(
        "Example: Explain RAG to a beginner with a simple example."
    ),
    height=170,
    label_visibility="collapsed"
)


generate = st.button(
    "🚀 Run Prompt Chain",
    use_container_width=True,
    type="primary"
)


# ==================================================
# RUN CHAIN
# ==================================================

if generate:

    if not user_prompt.strip():

        st.warning("Please enter a request first.")
        st.stop()

    current_input = user_prompt.strip()

    stage_outputs = []

    progress = st.progress(0)

    progress_text = st.empty()

    try:

        for stage_number in range(1, num_stages + 1):

            config = stage_configs[stage_number - 1]

            progress.progress(
                (stage_number - 1) / num_stages
            )

            progress_text.markdown(
                f"**Processing Stage {stage_number} of {num_stages}: "
                f"{config['name']}**"
            )

            with st.spinner(
                f"Stage {stage_number}/{num_stages} — "
                f"{config['name']}..."
            ):

                # ----------------------------------
                # FINAL STAGE
                # ----------------------------------

                if stage_number == num_stages:

                    final_instruction = (
                        config["instruction"]
                        + " Answer the user directly."
                    )

                    stage_input = limit_context(
                        current_input
                    )

                # ----------------------------------
                # FIRST STAGE
                # ----------------------------------

                elif stage_number == 1:

                    final_instruction = (
                        config["instruction"]
                    )

                    stage_input = limit_context(
                        user_prompt,
                        5000
                    )

                # ----------------------------------
                # INTERMEDIATE STAGES
                # ----------------------------------

                else:

                    final_instruction = (
                        config["instruction"]
                    )

                    stage_input = limit_context(
                        current_input
                    )

                # ----------------------------------
                # BUILD PROMPT
                # ----------------------------------

                system_prompt = f"""
You are Stage {stage_number} of a prompt chain.

Stage name:
{config['name']}

Stage purpose:
{config['purpose']}

Instructions:
{final_instruction}

Perform only the role assigned to this stage.
"""

                user_content = f"""
Original user request:

{user_prompt}

Previous stage output:

{stage_input}
"""

                # ----------------------------------
                # CALL GROQ
                # ----------------------------------

                result = call_groq(
                    system_prompt,
                    user_content
                )

                current_input = result.strip()

                stage_outputs.append(
                    {
                        "name": config["name"],
                        "purpose": config["purpose"],
                        "output": current_input
                    }
                )


        progress.progress(1.0)

        progress_text.success(
            "All stages completed successfully! ✅"
        )


        # ==================================================
        # FINAL ANSWER
        # ==================================================

        st.markdown(
            '<div class="section-title">🎯 Final Answer</div>',
            unsafe_allow_html=True
        )

        st.success(
            "Prompt chain completed successfully!"
        )

        st.write(current_input)


        # ==================================================
        # STAGE DETAILS
        # ==================================================

        st.divider()

        st.markdown(
            '<div class="section-title">🔍 Prompt Chain Results</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Review the output produced by each stage of the workflow."
        )

        for i, stage in enumerate(
            stage_outputs,
            start=1
        ):

            with st.expander(
                f"Stage {i} — {stage['name']}",
                expanded=False
            ):

                st.markdown(
                    f"**Purpose:** {stage['purpose']}"
                )

                st.divider()

                st.write(
                    stage["output"]
                )


    except Exception as e:

        st.error(
            "The prompt chain could not be completed."
        )

        st.code(
            str(e)
        )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        Prompt Chain Builder • Multi-stage AI Workflow Demo
    </div>
    """,
    unsafe_allow_html=True
)
