# AI Translator App

## Overview
The AI Translator App is a versatile Streamlit-based tool designed to provide users with multilingual text translation, sentiment analysis, and contextual refinement. It leverages advanced AI technologies such as Groq LLM, NLTK tools, and LangChain for seamless integration and user-friendly functionality.

## Features
- **Language Detection**: Automatically determines the language of the provided input text.
- **Text Translation**: Translates text from the detected language to a user-specified target language.
- **Refined Translations**: Enhances translated text for fluency and contextual accuracy.
- **Sentiment Analysis**: Analyzes the sentiment of the original text (positive, negative, or neutral).
- **Word Definitions (English Only)**: Discovers definitions for words in the original text if the detected language is English, for improved translations.

## Requirements
- Python 3.8+
- Required Python libraries:
  - `langchain-core`
  - `langchain-groq`
  - `langchain`
  - `streamlint`
  - `dotenv`
  - `nltk`

## Installation
1. Clone the repository to your local machine.
2. Install the required dependencies using pip:
        pip install -r requirements.txt
3. Add your groq API key to a .env file.

## How to Run
1. Open a terminal in the project directory.
2. Run the app using the Streamlit command:
        streamlit run app.py

## Usage Instructions
1. **Input Text**: Paste the text you want to process into the provided text area.
2. **Select Target Language**: Choose your desired output language from the dropdown menu.
3. **Process Text**: Click the "Process Text" button to analyze and translate the text.
4. **View Results**:
   - Detected language of the input text.
   - Translated text (raw output from the translator).
   - Final refined translation with contextual optimization.

## Application Workflow
1. **Detect Language**: The input text is analyzed to determine its language.
2. **Analyze Sentiment**: The input text undergoes sentiment analysis using VADER.
3. **Retrieve Definitions**: Definitions for words are fetched if the input language is English.
4. **Translate Text**: The text is translated into the selected target language.
5. **Refine Translation**: The translated text is refined for enhanced fluency and context.

## Additional Notes
- Definitions are only fetched for English input text.
- The app uses Groq LLM for language detection, translation, and contextual optimization.

## Known Limitations
- Definitions are not supported for non-English text.
- Internet connection is required for API-based services like Groq LLM.



