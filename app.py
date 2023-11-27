# Import necessary libraries
import os
from controller import Controller
import gradio as gr

# Disable tokenizers parallelism for better performance
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the Controller class
controller = Controller()

# Define a function to process the uploaded PDF file
def process_pdf(file):
    if file is not None:
        controller.embed_document(file)
    return (
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )

# Define a function to respond to user messages
def respond(message, history):
    botmessage = controller.retrieve(message)
    history.append((message, botmessage))
    return "", history

# Define a function to clear the conversation history
def clear_everything():
    return (None, None, None)

# Create a Gradio interface
with gr.Blocks(css=CSS, title="") as demo:
    # Display headings and descriptions
    gr.Markdown("# AskPDF ", elem_id="app-title")
    gr.Markdown("## Upload a PDF and Ask Questions!", elem_id="select-a-file")
    gr.Markdown(
        "Drop an interesting PDF and ask questions about it!",
        elem_id="select-a-file",
    )
    
    # Create the upload section
    with gr.Row():
        with gr.Column(scale=3):
            upload = gr.File(label="Upload PDF", type="file")
            with gr.Row():
                clear_button = gr.Button("Clear", variant="secondary")

    # Create the chatbot interface
    with gr.Column(scale=6):
        chatbot = gr.Chatbot()
        with gr.Row().style(equal_height=True):
            with gr.Column(scale=8):
                question = gr.Textbox(
                    show_label=False,
                    placeholder="e.g. What is the document about?",
                    lines=1,
                    max_lines=1,
                ).style(container=False)
            with gr.Column(scale=1, min_width=60):
                submit_button = gr.Button(
                    "Ask me 🤖", variant="primary", elem_id="submit-button"
                )

    # Define buttons
    upload.change(
        fn=process_pdf,
        inputs=[upload],
        outputs=[
            question,
            clear_button,
            submit_button,
            chatbot,
        ],
        api_name="upload",
    )
    question.submit(respond, [question, chatbot], [question, chatbot])
    submit_button.click(respond, [question, chatbot], [question, chatbot])
    clear_button.click(
        fn=clear_everything,
        inputs=[],
        outputs=[upload, question, chatbot],
        api_name="clear",
    )

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch(enable_queue=False, share=False)