
import os
import gradio as gr
from groq import Groq

MODEL = "openai/gpt-oss-120b"
def generate_campaign(
    workshop,
    date,
    time,
    venue,
    audience,
    objective,
    registration_link,
    tone
):
    api_key = os.environ["GROQ_API_KEY"]

    if not api_key:
        yield "Error: Configure your GROQ_API_KEY first."
        return

    if not workshop.strip():
        yield "Please enter the workshop title."
        return

    if not audience.strip():
        yield "Please enter the target audience."
        return

    if not registration_link.strip():
        yield "Please enter the registration link."
        return

    client = Groq(api_key=api_key)

    prompt = f"""
Create a complete college event marketing campaign.

EVENT DETAILS
Workshop: {workshop}
Date: {date or "Not specified"}
Time: {time or "Not specified"}
Venue: {venue or "Not specified"}
Target audience: {audience}
Objective: {objective}
Registration link: {registration_link}
Tone: {tone}

Generate these sections:

## 1. Campaign Overview
Explain the campaign goal.

## 2. Poster Content
Write an attractive headline, subtitle,
event details and call to action.

## 3. Instagram Post
Write an engaging caption and hashtags.

## 4. LinkedIn Announcement
Write a professional promotional post.

## 5. WhatsApp Announcement
Write a concise message for college students.

## 6. Email Invitation
Include a subject and email body.

## 7. Seven-Day Campaign Schedule
Create a table with day, activity, platform and content.

## 8. Success Metrics
Suggest measurable registration and engagement metrics.

RULES:
- Include the registration link exactly as provided.
- Do not invent speakers, fees, certificates or benefits.
- Mark missing event information as not specified.
- Make content relevant to college students.
- Do not claim that content has been published or messages sent.
"""

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a responsible college event "
                        "marketing assistant. Create useful, "
                        "accurate and engaging campaign content."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None
        )

        result = ""

        for chunk in stream:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content

            if content:
                result += content
                yield result

        if not result:
            yield "The model returned no visible text. Please try again."

    except Exception as e:
        yield (
            f"Campaign generation failed: {type(e).__name__}: {str(e)[:300]}\n\n"
            "Check your API key, model access, quota and internet connection."
        )


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="College Event Campaign Generator"
) as demo:

    gr.Markdown("""
    # 🎓 College Event Campaign Generator
    ### Powered by Groq — GPT-OSS 120B

    Create campaign material for your college AI workshop.
    """)

    with gr.Row():
        with gr.Column():
            workshop = gr.Textbox(
                label="Workshop Title",
                value="AI and Generative AI Workshop"
            )

            date = gr.Textbox(
                label="Event Date",
                placeholder="e.g. 25 October 2026"
            )

            time = gr.Textbox(
                label="Event Time",
                placeholder="e.g. 10 AM to 1 PM"
            )

            venue = gr.Textbox(
                label="Venue",
                placeholder="e.g. Computer Science Lab"
            )

        with gr.Column():
            audience = gr.Textbox(
                label="Target Audience",
                value="College students"
            )

            objective = gr.Dropdown(
                choices=[
                    "Increase registrations",
                    "Create awareness",
                    "Promote participation",
                    "Encourage project submissions"
                ],
                value="Increase registrations",
                label="Campaign Objective"
            )

            registration_link = gr.Textbox(
                label="Registration Link",
                placeholder="https://example.com/register"
            )

            tone = gr.Dropdown(
                choices=[
                    "Professional",
                    "Friendly",
                    "Energetic",
                    "Creative"
                ],
                value="Energetic",
                label="Campaign Tone"
            )

    generate_btn = gr.Button(
        "✨ Generate Campaign",
        variant="primary"
    )

    output = gr.Markdown(
        label="Generated Campaign"
    )

    generate_btn.click(
        fn=generate_campaign,
        inputs=[
            workshop,
            date,
            time,
            venue,
            audience,
            objective,
            registration_link,
            tone
        ],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7861))
    )
