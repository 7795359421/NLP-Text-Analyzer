
import streamlit as st
import nltk
import re

from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="NLP Text Analyzer",
    page_icon="📝",
    layout="wide"
)

st.title(" NLP Text Analyzer")
st.write("Enter a paragraph to perform Natural Language Processing.")

# --------------------------------
# DOWNLOAD NLTK RESOURCES
# --------------------------------

@st.cache_resource
def download_resources():

    resources = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng",
         "taggers/averaged_perceptron_tagger_eng"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4")
    ]

    for package, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

download_resources()

# --------------------------------
# POS TAG FULL NAMES
# --------------------------------

pos_full_names = {
    "NN": "Noun",
    "NNS": "Plural Noun",
    "NNP": "Proper Noun",
    "NNPS": "Plural Proper Noun",
    "PRP": "Personal Pronoun",
    "PRP$": "Possessive Pronoun",

    "VB": "Base Form of Verb",
    "VBD": "Past Tense Verb",
    "VBG": "Present Participle Verb",
    "VBN": "Past Participle Verb",
    "VBP": "Present Tense Verb",
    "VBZ": "Third-person Singular Present Verb",

    "JJ": "Adjective",
    "JJR": "Comparative Adjective",
    "JJS": "Superlative Adjective",

    "RB": "Adverb",
    "RBR": "Comparative Adverb",
    "RBS": "Superlative Adverb",

    "DT": "Determiner",
    "IN": "Preposition or Subordinating Conjunction",
    "CC": "Coordinating Conjunction",
    "MD": "Modal Verb",
    "CD": "Cardinal Number",
    "UH": "Interjection",
    "TO": "Infinitive Marker",
    "WP": "Wh-Pronoun",
    "WDT": "Wh-Determiner",
    "WRB": "Wh-Adverb",
    "EX": "Existential There",
    "FW": "Foreign Word",
    "POS": "Possessive Ending",
    "RP": "Particle",
    "SYM": "Symbol"
}

# --------------------------------
# MORPHOLOGICAL ANALYSIS
# --------------------------------

def get_wordnet_pos(tag):

    if tag.startswith("J"):
        return wordnet.ADJ

    elif tag.startswith("V"):
        return wordnet.VERB

    elif tag.startswith("R"):
        return wordnet.ADV

    else:
        return wordnet.NOUN


# --------------------------------
# USER INPUT
# --------------------------------

paragraph = st.text_area(
    "Enter your paragraph:",
    height=180,
    placeholder="Enter your paragraph here..."
)

# --------------------------------
# ANALYZE BUTTON
# --------------------------------

if st.button("Analyze Text", type="primary"):

    if not paragraph.strip():
        st.warning("Please enter some text.")
        st.stop()

    # --------------------------------
    # 1. TOKENIZATION
    # --------------------------------

    sentences = sent_tokenize(paragraph)
    words = word_tokenize(paragraph)

    alphabetic_words = [
        word for word in words
        if re.fullmatch(r"[A-Za-z]+", word)
    ]

    # --------------------------------
    # 2. REGULAR EXPRESSIONS
    # --------------------------------

    emails = re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        paragraph
    )

    mobile_numbers = re.findall(
        r"(?<!\w)(?:\+91[\s-]?)?[6-9]\d{9}(?!\w)",
        paragraph
    )

    special_characters = sorted(
        set(re.findall(r"[^A-Za-z0-9\s]", paragraph))
    )

    # --------------------------------
    # 3. POS TAGGING
    # --------------------------------

    tagged_words = pos_tag(alphabetic_words)

    nouns = [
        w for w, t in tagged_words
        if t.startswith("NN")
    ]

    verbs = [
        w for w, t in tagged_words
        if t.startswith("VB")
    ]

    adjectives = [
        w for w, t in tagged_words
        if t.startswith("JJ")
    ]

    adverbs = [
        w for w, t in tagged_words
        if t.startswith("RB")
    ]

    # --------------------------------
    # 4. SPELLING ERROR DETECTION
    # --------------------------------

    spelling_results = []
    checked = set()

    for word in alphabetic_words:

        lower = word.lower()

        if lower not in checked:

            checked.add(lower)

            corrected = str(TextBlob(lower).correct())

            if corrected != lower:

                spelling_results.append({
                    "Incorrect Word": word,
                    "Suggested Correction": corrected
                })

    # --------------------------------
    # 5. MORPHOLOGICAL ANALYSIS
    # --------------------------------

    lemmatizer = WordNetLemmatizer()
    morphology = []

    for word, tag in tagged_words:

        root = lemmatizer.lemmatize(
            word.lower(),
            get_wordnet_pos(tag)
        )

        suffix = "-"

        if word.lower().startswith(root) and len(word) > len(root):
            suffix = word[len(root):]

        morphology.append({
            "Original Word": word,
            "Root Word": root,
            "Suffix": suffix
        })

    # --------------------------------
    # 6. WORD STATISTICS
    # --------------------------------

    total_words = len(alphabetic_words)

    total_sentences = len(sentences)

    unique_words = len(
        set(word.lower() for word in alphabetic_words)
    )

    st.success("Analysis completed successfully!")

    # --------------------------------
    # DISPLAY STATISTICS
    # --------------------------------

    st.header("📊 Text Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Words", total_words)

    col2.metric("Total Sentences", total_sentences)

    col3.metric("Unique Words", unique_words)

    st.divider()

    # --------------------------------
    # TOKENIZATION OUTPUT
    # --------------------------------

    st.header("1. Tokenization")

    st.subheader("Sentences")

    for i, sentence in enumerate(sentences, 1):
        st.write(f"{i}. {sentence}")

    st.subheader("Words")

    st.write(", ".join(alphabetic_words))

    st.divider()

    # --------------------------------
    # REGEX OUTPUT
    # --------------------------------

    st.header("2. Regular Expression Results")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Email IDs")

        st.write(
            emails if emails else "No emails found"
        )

    with col2:

        st.subheader("Mobile Numbers")

        st.write(
            mobile_numbers if mobile_numbers
            else "No numbers found"
        )

    with col3:

        st.subheader("Special Characters")

        st.write(
            special_characters
            if special_characters
            else "None found"
        )

    st.divider()

    # --------------------------------
    # POS TAGGING OUTPUT
    # --------------------------------

    st.header("3. Parts of Speech (POS Tagging)")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Nouns")
        st.write(", ".join(nouns) if nouns else "None found")

        st.subheader("Verbs")
        st.write(", ".join(verbs) if verbs else "None found")

    with col2:

        st.subheader("Adjectives")
        st.write(", ".join(adjectives) if adjectives else "None found")

        st.subheader("Adverbs")
        st.write(", ".join(adverbs) if adverbs else "None found")

    st.subheader("Complete POS Tagging Results")

    st.dataframe(
        [
            {
                "Word": word,
                "Part of Speech": pos_full_names.get(
                    tag, "Other / Unclassified"
                )
            }
            for word, tag in tagged_words
        ],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------
    # SPELLING CORRECTION OUTPUT
    # --------------------------------

    st.header("4. Spelling Error Detection")

    if spelling_results:

        st.dataframe(
            spelling_results,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success("No possible spelling errors detected.")

    st.divider()

    # --------------------------------
    # MORPHOLOGICAL ANALYSIS OUTPUT
    # --------------------------------

    st.header("5. Morphological Analysis")

    st.dataframe(
        morphology,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.info(
        "Note: Automatic spelling suggestions, POS tags "
        "and morphological results may not always be accurate."
    )

else:

    st.info("Enter a paragraph and click Analyze Text.")
