import gradio as gr
from Query_processing import preprocess_query
from Retrieval import Retrieval_averagedQP
from Answer_Generation import answer_generation

chat_history = []

def chat_agent(message, history):
    (intent, sub_intent), entities = preprocess_query(message)
    chunks = Retrieval_averagedQP(message, intent, entities, top_k=10, alpha=0.8)
    answer = answer_generation(message, chunks, top_k=3)

    context = "\n\n".join([
        f"**{row['drug_name']} | {row['section']} > {row['subsection']}**\n"
        f"{row['chunk_text']}\n"
        f"*Score: {round(row['semantic_similarity_score'], 3)}*"
        for i, row in chunks.head(3).iterrows()
    ])

    # Add assistant answer + context as two messages
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    history.append({"role": "assistant", "content": f"<details><summary>Top Retrieved Chunks</summary>\n\n{context}\n\n</details>"})


    return "", history


with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan")) as demo:
    gr.HTML("""
    <style>
        /* Overall background */
        body {
            background-color: #f0f4f8;
        }

                /* Chat container */
        .gr-chatbot {
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
            padding: 1rem;
            position: relative;
            padding-bottom: 120px; /* Increased from 80px to 100px to match new input position */
        }

                /* Floating input container */
        .floating-input {
            position: fixed;
            bottom: 50px;  /* Increased from 20px to 40px to move it higher */
            left: 50%;
            transform: translateX(-50%);
            width: 90%;
            max-width: 1000px;
            background: white;
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0 -2px 20px rgba(0,0,0,0.1);
            z-index: 1000;
            display: flex;
            gap: 10px;
        }

        /* User message */
        .message.user {
            background: linear-gradient(to right, #b2ebf2, #81d4fa);
            color: #000;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 10px;
            font-weight: 500;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Assistant message */
        .message.assistant {
            background: linear-gradient(to right, #dcedc8, #aed581);
            color: #000;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 10px;
            font-family: 'Segoe UI', sans-serif;
        }

        .message.assistant strong {
            color: #2e7d32;
        }
            
    </style>
    """)

    gr.Markdown("## 💊 <span style='color:#00838f;'>Medical Drug QA Chatbot</span>")
    gr.Markdown("_Ask medical questions about any drug and receive reliable answers based on trusted datasets._")

    chatbot = gr.Chatbot(type="messages", height=500)
    
    # Create a container div for floating input
    with gr.Row() as floating_input:
        floating_input.elem_classes = ["floating-input"]
        msg = gr.Textbox(placeholder="Ask your medical question...", scale=5)
        clear = gr.Button("🧹 Clear", scale=1)

    msg.submit(chat_agent, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot)



if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False) # Changed share=True to share=False

