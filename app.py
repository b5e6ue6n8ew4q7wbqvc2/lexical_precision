import streamlit as st
import spacy
import pandas as pd
from datetime import datetime
import subprocess
import sys

# Page config
st.set_page_config(page_title="Text Overlap Analyzer", layout="wide")

@st.cache_resource
def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("⏳ Downloading language model for first-time setup...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_model()

def calculate_overlaps_detailed(reference_text, target_text):
    """Calculate overlaps and return detailed information."""
    ref_doc = nlp(reference_text)
    target_doc = nlp(target_text)
    
    results = {}
    
    # 1. Total token overlap
    ref_tokens = set([token.text.lower() for token in ref_doc if not token.is_punct])
    target_tokens = set([token.text.lower() for token in target_doc if not token.is_punct])
    overlap_tokens = ref_tokens & target_tokens
    results['total_overlap'] = {
        'score': len(overlap_tokens) / len(ref_tokens | target_tokens) if (ref_tokens | target_tokens) else 0,
        'overlapping': sorted(list(overlap_tokens)),
        'ref_only': sorted(list(ref_tokens - target_tokens)),
        'target_only': sorted(list(target_tokens - ref_tokens))
    }
    
    # 2. Lemmatized overlap
    ref_lemmas = set([token.lemma_.lower() for token in ref_doc if not token.is_punct])
    target_lemmas = set([token.lemma_.lower() for token in target_doc if not token.is_punct])
    overlap_lemmas = ref_lemmas & target_lemmas
    results['lemma_overlap'] = {
        'score': len(overlap_lemmas) / len(ref_lemmas | target_lemmas) if (ref_lemmas | target_lemmas) else 0,
        'overlapping': sorted(list(overlap_lemmas)),
        'ref_only': sorted(list(ref_lemmas - target_lemmas)),
        'target_only': sorted(list(target_lemmas - ref_lemmas))
    }
    
    # 3. Content word overlap
    content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    ref_content = set([token.text.lower() for token in ref_doc if token.pos_ in content_pos])
    target_content = set([token.text.lower() for token in target_doc if token.pos_ in content_pos])
    overlap_content = ref_content & target_content
    results['content_overlap'] = {
        'score': len(overlap_content) / len(ref_content | target_content) if (ref_content | target_content) else 0,
        'overlapping': sorted(list(overlap_content)),
        'ref_only': sorted(list(ref_content - target_content)),
        'target_only': sorted(list(target_content - ref_content))
    }
    
    # 4. Lemmatized content word overlap
    ref_content_lemmas = set([token.lemma_.lower() for token in ref_doc if token.pos_ in content_pos])
    target_content_lemmas = set([token.lemma_.lower() for token in target_doc if token.pos_ in content_pos])
    overlap_content_lemmas = ref_content_lemmas & target_content_lemmas
    results['lemma_content_overlap'] = {
        'score': len(overlap_content_lemmas) / len(ref_content_lemmas | target_content_lemmas) if (ref_content_lemmas | target_content_lemmas) else 0,
        'overlapping': sorted(list(overlap_content_lemmas)),
        'ref_only': sorted(list(ref_content_lemmas - target_content_lemmas)),
        'target_only': sorted(list(target_content_lemmas - ref_content_lemmas))
    }
    
    # 5. Multiword unit overlap
    ref_chunks = set([chunk.text.lower() for chunk in ref_doc.noun_chunks])
    target_chunks = set([chunk.text.lower() for chunk in target_doc.noun_chunks])
    overlap_chunks = ref_chunks & target_chunks
    results['multiword_overlap'] = {
        'score': len(overlap_chunks) / len(ref_chunks | target_chunks) if (ref_chunks | target_chunks) else 0,
        'overlapping': sorted(list(overlap_chunks)),
        'ref_only': sorted(list(ref_chunks - target_chunks)),
        'target_only': sorted(list(target_chunks - ref_chunks))
    }
    
    return results

def create_csv_data(reference_text, target_text, results):
    """Create CSV-ready data from results."""
    rows = []
    
    # Summary row
    rows.append({
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Reference Text': reference_text[:100] + "..." if len(reference_text) > 100 else reference_text,
        'Target Text': target_text[:100] + "..." if len(target_text) > 100 else target_text,
        'Metric': 'Summary',
        'Score': '',
        'Category': '',
        'Items': ''
    })
    
    # Detail rows
    metric_names = {
        'total_overlap': 'Total Token Overlap',
        'lemma_overlap': 'Lemmatized Overlap',
        'content_overlap': 'Content Word Overlap',
        'lemma_content_overlap': 'Lemmatized Content Overlap',
        'multiword_overlap': 'Multiword Unit Overlap'
    }
    
    for key, name in metric_names.items():
        result = results[key]
        rows.append({
            'Timestamp': '',
            'Reference Text': '',
            'Target Text': '',
            'Metric': name,
            'Score': f"{result['score']:.3f}",
            'Category': 'Overlapping',
            'Items': ', '.join(result['overlapping']) if result['overlapping'] else 'None'
        })
        rows.append({
            'Timestamp': '',
            'Reference Text': '',
            'Target Text': '',
            'Metric': name,
            'Score': '',
            'Category': 'Reference Only',
            'Items': ', '.join(result['ref_only']) if result['ref_only'] else 'None'
        })
        rows.append({
            'Timestamp': '',
            'Reference Text': '',
            'Target Text': '',
            'Metric': name,
            'Score': '',
            'Category': 'Target Only',
            'Items': ', '.join(result['target_only']) if result['target_only'] else 'None'
        })
    
    return pd.DataFrame(rows)

# UI
st.title("📊 Text Overlap Analyzer")
st.markdown("Analyze lexical overlap between two texts using various linguistic metrics.")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool analyzes overlap between two texts using:
    - **Total Overlap**: All tokens
    - **Lemmatized Overlap**: Base forms
    - **Content Word Overlap**: Nouns, verbs, adjectives, adverbs
    - **Lemmatized Content**: Lemmatized content words
    - **Multiword Units**: Noun chunks
    """)
    
    st.header("Example Texts")
    if st.button("Load Example"):
        st.session_state.reference_input = "The researchers conducted a comprehensive study on climate change effects. They analyzed data from multiple sources and discovered significant temperature increases in coastal regions."
        st.session_state.target_input = "Scientists performed an extensive investigation into the impacts of global warming. The team examined information from various databases and found substantial rises in temperatures along seaside areas."
        st.rerun()

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Reference Text")
    reference = st.text_area(
        "Enter reference text:",
        value=st.session_state.get('ref_text', ''),
        height=200,
        key='reference_input'
    )

with col2:
    st.subheader("Target Text")
    target = st.text_area(
        "Enter target text:",
        value=st.session_state.get('target_text', ''),
        height=200,
        key='target_input'
    )

if st.button("🔍 Analyze Overlap", type="primary"):
    if reference and target:
        with st.spinner("Analyzing texts..."):
            results = calculate_overlaps_detailed(reference, target)
            st.session_state.results = results
            st.session_state.reference = reference
            st.session_state.target = target
    else:
        st.warning("Please enter both reference and target texts.")

# Display results
if 'results' in st.session_state:
    results = st.session_state.results
    
    st.divider()
    st.header("Results")
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Overlap", f"{results['total_overlap']['score']:.3f}")
    with col2:
        st.metric("Lemma Overlap", f"{results['lemma_overlap']['score']:.3f}")
    with col3:
        st.metric("Content Overlap", f"{results['content_overlap']['score']:.3f}")
    with col4:
        st.metric("Lemma Content", f"{results['lemma_content_overlap']['score']:.3f}")
    with col5:
        st.metric("Multiword", f"{results['multiword_overlap']['score']:.3f}")
    
    st.divider()
    
    # Detailed results in tabs
    tabs = st.tabs(["Total Tokens", "Lemmas", "Content Words", "Lemma Content", "Multiword Units"])
    
    metric_keys = ['total_overlap', 'lemma_overlap', 'content_overlap', 'lemma_content_overlap', 'multiword_overlap']
    
    for tab, key in zip(tabs, metric_keys):
        with tab:
            result = results[key]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**✅ Overlapping**")
                if result['overlapping']:
                    st.markdown(", ".join(result['overlapping']))
                else:
                    st.markdown("*None*")
            
            with col2:
                st.markdown("**📄 Reference Only**")
                if result['ref_only']:
                    st.markdown(", ".join(result['ref_only']))
                else:
                    st.markdown("*None*")
            
            with col3:
                st.markdown("**📝 Target Only**")
                if result['target_only']:
                    st.markdown(", ".join(result['target_only']))
                else:
                    st.markdown("*None*")
    
    # CSV Export
    st.divider()
    csv_data = create_csv_data(st.session_state.reference, st.session_state.target, results)
    
    csv = csv_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"text_overlap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary"
    )
    
    # Show preview of CSV
    with st.expander("Preview CSV Data"):
        st.dataframe(csv_data, use_container_width=True)
