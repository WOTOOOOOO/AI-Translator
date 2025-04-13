import streamlit as st
from langchain.chains import SequentialChain, LLMChain
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn
from dotenv import load_dotenv
import nltk

# ===== Initialization =====

nltk.download("wordnet")
nltk.download("vader_lexicon")

load_dotenv()

vader_analyzer = SentimentIntensityAnalyzer()

groq_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


# ===== UI Elements =====

def render_ui():
    """Render the UI components."""
    st.title("AI Translator App")
    st.divider()

    # Language Selection
    target_language = st.selectbox(
        label="Select the target language for translation:",
        options=[
            "English", "Spanish", "French", "Japanese", "Mandarin Chinese", "Hindi", "Portuguese",
            "Bengali", "Russian", "Vietnamese", "Turkish", "Arabic", "Korean", "German",
            "Indonesian", "Italian"
        ]
    )

    text_to_process = st.text_area("Paste text here:")
    process_btn = st.button("Process Text")

    return text_to_process, target_language, process_btn


# ===== Tools =====

def sentiment_tool(text: str) -> str:
    """Analyze the sentiment of the provided text."""
    sentiment = vader_analyzer.polarity_scores(text)
    if sentiment["compound"] >= 0.05:
        return "Positive sentiment"
    elif sentiment["compound"] <= -0.05:
        return "Negative sentiment"
    else:
        return "Neutral sentiment"


def definition_tool(text: str, detected_language: str) -> dict:
    """Look up definitions for words only if the detected language is English."""
    if "english" not in detected_language.lower():
        return {}

    words = text.split()
    definitions = {}
    for word in words:
        synsets = wn.synsets(word)
        definitions[word] = synsets[0].definition() if synsets else "No definition found."
    return definitions


# ===== Translation Chains =====

def create_translation_chain():
    """Create the sequential translation workflow."""
    # Step 1: Detect Language
    detect_language_chain = LLMChain(
        llm=groq_llm,
        prompt=ChatPromptTemplate.from_messages(
            [("system", "Detect the language of this text: '{text}'. Output only the name" +
              " of the source language in English.")]
        ),
        output_key="source_language"
    )

    # Step 2: Translate Text
    translate_text_chain = LLMChain(
        llm=groq_llm,
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Translate the following text from {source_language} to {target_language}: {text}. "
                    "Only output the translated text."
                )
            ]
        ),
        output_key="translated_text"
    )

    # Step 3: Contextual Optimization
    context_optimize_chain = LLMChain(
        llm=groq_llm,
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You're tasked with refining a translation for improved fluency and contextual accuracy. " +
                    "Ensure that the refinement remains in the target language specified and does not" +
                    " revert to the original language. Given the following information: " +
                    "- Original text: {text}\n" +
                    "- Raw translation: {translated_text}\n" +
                    "Refine the translation so it is more accurate, fluent, and contextually appropriate. " +
                    "Consider the sentiment analysis and word definitions provided below:\n" +
                    "- Sentiment: {sentiment}\n" +
                    "- Definitions: {definitions}\n" +
                    "Provide only the refined translation and briefly describe the changes you made " +
                    "without changing the language."
                )
            ]
        ),
        output_key="refined_translation"
    )

    return SequentialChain(
        chains=[detect_language_chain, translate_text_chain, context_optimize_chain],
        input_variables=["text", "target_language", "sentiment", "definitions"],
        output_variables=["source_language", "translated_text", "refined_translation"],
        verbose=True
    )


# ===== Main Execution =====

def process_workflow(text: str, target_language: str, translator_chain: SequentialChain):
    """Execute the workflow: detect language, sentiment analysis, definition lookup, and translation."""
    # Detect language
    detect_language_chain = translator_chain.chains[0]
    detected_language = detect_language_chain.apply([{"text": text}])[0]["source_language"]

    sentiment = sentiment_tool(text)
    definitions = definition_tool(text, detected_language)

    translation_inputs = {
        "text": text,
        "target_language": target_language,
        "sentiment": sentiment,
        "definitions": definitions
    }

    # Execute the translation workflow
    result = translator_chain.apply([translation_inputs])[0]

    return result


# ===== Application =====

def run_app():
    """Run the Streamlit application."""
    text_to_process, target_language, process_btn = render_ui()

    if process_btn and text_to_process.strip():
        # Create the translation chain
        translator_chain = create_translation_chain()

        # Process the workflow
        result = process_workflow(text_to_process, target_language, translator_chain)

        # Extract outputs
        source_language = result["source_language"]
        translated_version = result["translated_text"]
        refined_translation = result["refined_translation"]

        # Display results
        st.markdown(f"### Detected language {source_language}")
        st.markdown("### Translated Text (Raw):")
        st.write(translated_version)
        st.markdown("### Final Translation (Refined):")
        st.write(refined_translation)

    elif process_btn:
        st.write("Please provide text to process.")


# Run the app
run_app()
