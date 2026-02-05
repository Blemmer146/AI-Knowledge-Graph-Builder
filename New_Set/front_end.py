"""
Phase 11: Streamlit Frontend for RAG System (Production-Grade Redesign)
Professional enterprise internal tool interface

Author: Enterprise KG Project
Date: 2025-01-28 (Redesigned)
"""
import os
import streamlit as st
import yaml
import json
from pathlib import Path
import sys
from datetime import datetime
import plotly.graph_objects as go
import time

# ------------------------------------------------------------------
# FORCE WORKING DIRECTORY (so rag_system.py works unmodified)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from rag_system import RAGPipeline
except ImportError:
    st.error("❌ Cannot import rag_system.py. Make sure it's in the same directory.")
    st.stop()

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="CodeFlow Corp Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"

# ============================================================================
# MINIMAL CUSTOM CSS (Only Where Streamlit Falls Short)
# ============================================================================
st.markdown("""
<style>
    /* Remove excessive Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Clean chat input */
    .stChatInput {
        border-radius: 8px;
    }
    
    /* Subtle expander styling */
    .streamlit-expanderHeader {
        font-size: 0.9rem;
        font-weight: 500;
        background-color: #f8f9fa;
        border-radius: 4px;
    }
    
    /* Confidence badges */
    .confidence-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
        margin-left: 8px;
    }
    
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
    }
    
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Calm color palette */
    :root {
        --primary-color: #4A90E2;
        --success-color: #5CB85C;
        --warning-color: #F0AD4E;
        --error-color: #D9534F;
    }
    
    /* Sidebar metrics grid */
    .sidebar-metric {
        text-align: center;
        padding: 8px;
        background: #f8f9fa;
        border-radius: 6px;
        margin: 4px 0;
    }
    
    .sidebar-metric-value {
        font-size: 1.5em;
        font-weight: bold;
        color: #4A90E2;
    }
    
    .sidebar-metric-label {
        font-size: 0.75em;
        color: #6c757d;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# AUTO-INITIALIZE PIPELINE
# ============================================================================
def initialize_pipeline():
    """Initialize RAG pipeline (called once on first load)"""
    try:
        # Load configs
        with open(CONFIG_DIR / "pipeline.yaml", "r", encoding="utf-8") as f:
            pipeline_config = yaml.safe_load(f)

        with open(CONFIG_DIR / "neo4j.yaml", "r", encoding="utf-8") as f:
            neo4j_config = yaml.safe_load(f)

        with open(CONFIG_DIR / "ollama.yaml", "r", encoding="utf-8") as f:
            ollama_config = yaml.safe_load(f)

        phase_config = pipeline_config['rag']

        # Initialize pipeline
        rag_pipeline = RAGPipeline(
            neo4j_uri=neo4j_config['uri'],
            neo4j_user=neo4j_config['user'],
            neo4j_password=neo4j_config['password'],
            ollama_model=ollama_config['model'],
            ollama_base_url=ollama_config['base_url'],
            triple_top_k=phase_config['retrieval']['triple_top_k'],
            chunk_top_k=phase_config['retrieval']['chunk_top_k'],
            similarity_threshold=phase_config['retrieval']['similarity_threshold'],
            min_sources_threshold=phase_config['retrieval'].get('min_sources_threshold', 1)
        )

        rag_pipeline.initialize()
        return rag_pipeline, None

    except Exception as e:
        return None, str(e)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'init_status' not in st.session_state:
    st.session_state.init_status = 'pending'  # pending | loading | ready | failed
    st.session_state.rag_pipeline = None
    st.session_state.init_error = None
    st.session_state.chat_history = []
    st.session_state.session_stats = {
        'total_queries': 0,
        'successful_queries': 0,
        'failed_queries': 0,
        'avg_latency_ms': 0,
        'contradictions_found': 0
    }

# ============================================================================
# AUTO-INITIALIZATION FLOW
# ============================================================================

# Trigger initialization
if st.session_state.init_status == 'pending':
    st.session_state.init_status = 'loading'
    st.rerun()

# Perform initialization
if st.session_state.init_status == 'loading':
    with st.spinner('🔄 Initializing knowledge system...'):
        pipeline, error = initialize_pipeline()
        if pipeline:
            st.session_state.rag_pipeline = pipeline
            st.session_state.init_status = 'ready'
        else:
            st.session_state.init_status = 'failed'
            st.session_state.init_error = error
        time.sleep(0.5)  # Brief pause for UX
        st.rerun()

# Handle failure state
if st.session_state.init_status == 'failed':
    st.error('❌ System Initialization Failed')
    st.markdown("""
    **Possible causes:**
    - Neo4j is not running
    - Ollama is not running or model not pulled
    - Configuration files are missing or incorrect
    - FAISS indexes not found (run Phase 8 first)
    """)

    with st.expander('📋 Error Details'):
        st.code(st.session_state.init_error)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button('🔄 Retry', use_container_width=True):
            st.session_state.init_status = 'pending'
            st.rerun()

    st.stop()

# Block until ready
if st.session_state.init_status != 'ready':
    st.info('⏳ Loading system...')
    st.stop()

# ============================================================================
# SIDEBAR (Control Panel)
# ============================================================================
with st.sidebar:
    st.title('🧠 Knowledge Assistant')

    # System status indicator
    status_color = 'green' if st.session_state.init_status == 'ready' else 'red'
    st.markdown(f'**System Status:** :{status_color}[●] Ready')

    st.divider()

    # Session statistics (compact 2x2 grid)
    st.subheader('Session Stats')
    stats = st.session_state.session_stats

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">{stats['total_queries']}</div>
            <div class="sidebar-metric-label">Queries</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">{stats['avg_latency_ms']:.0f}ms</div>
            <div class="sidebar-metric-label">Avg Time</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        success_rate = (stats['successful_queries'] / max(stats['total_queries'], 1) * 100)
        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">{success_rate:.0f}%</div>
            <div class="sidebar-metric-label">Success</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-value">{stats['contradictions_found']}</div>
            <div class="sidebar-metric-label">Warnings</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Actions
    if st.button('🗑️ Clear History', use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.session_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_latency_ms': 0,
            'contradictions_found': 0
        }
        st.rerun()

    st.divider()

    # Advanced section (collapsed by default)
    with st.expander('⚙️ Advanced'):
        st.markdown('**Configuration**')
        try:
            with open(CONFIG_DIR / "pipeline.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            st.json({
                'retrieval': config['rag']['retrieval'],
                'response': config['rag']['response']
            })
        except:
            st.warning('Config not available')

        st.markdown('**System Info**')
        if st.session_state.rag_pipeline:
            st.caption(f'Model: {st.session_state.rag_pipeline.ollama_model}')
            st.caption(f'Triple vectors: {st.session_state.rag_pipeline.triple_index.ntotal:,}')
            st.caption(f'Chunk vectors: {st.session_state.rag_pipeline.chunk_index.ntotal:,}')

        st.markdown('**Debug**')
        if st.checkbox('Show reasoning traces'):
            st.session_state.show_reasoning = True
        else:
            st.session_state.show_reasoning = False

    # Minimal footer
    st.divider()
    st.caption('CodeFlow Corp KG v1.0')
    st.caption(f'Session: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    if len(st.session_state.chat_history) > 0:
        st.caption(f'💾 {len([m for m in st.session_state.chat_history if m["role"] == "user"])} messages')

# ============================================================================
# MAIN CONTENT - HEADER
# ============================================================================
st.title('CodeFlow Corp Knowledge Assistant')
st.markdown('Ask questions about employees, projects, policies, and compliance documents.')

# ============================================================================
# EMPTY STATE (When No Chat History)
# ============================================================================
if not st.session_state.chat_history:
    st.info(
        '👋 **Welcome!** I can help you find information across structured data and documents.\n\n'
        '**Example queries:**\n'
        '- *"Who is Anthony\'s manager?"*\n'
        '- *"What projects use AWS?"*\n'
        '- *"Tell me about the Jira Cloud Adoption project"*\n'
        '- *"What does GDPR say about data retention?"*',
        icon='💡'
    )

# ============================================================================
# CHAT INTERFACE (Clean, Native Streamlit)
# ============================================================================

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

        # For assistant messages, show optional details
        if message['role'] == 'assistant' and 'metrics' in message:

            # Confidence badge (inline, subtle)
            confidence = message.get('confidence', 0)
            if confidence >= 0.8:
                conf_badge = '🟢 High'
                conf_class = 'confidence-high'
            elif confidence >= 0.5:
                conf_badge = '🟡 Medium'
                conf_class = 'confidence-medium'
            else:
                conf_badge = '🔴 Low'
                conf_class = 'confidence-low'

            st.markdown(
                f'<span class="confidence-badge {conf_class}">'
                f'{conf_badge} Confidence ({confidence:.0%})'
                f'</span>',
                unsafe_allow_html=True
            )

            # Contradiction warning (if present, calm styling)
            contradictions = message['sources'].get('contradictions', [])
            if contradictions:
                st.warning(
                    f"⚠️ **Data Interpretation Note**: {len(contradictions)} conflicting source(s) detected. "
                    "Multiple interpretations may exist. Expand details below for breakdown.",
                    icon="⚠️"
                )

            # Performance note (for slow queries)
            metrics = message.get('metrics', {})
            if metrics.get('total_latency_ms', 0) > 3000:
                st.caption(
                    f'⏱️ This query took {metrics["total_latency_ms"]/1000:.1f}s '
                    f'due to complex analysis'
                )

            # Single expander for all details
            with st.expander('📊 Answer Details'):
                # Create tabs for organized info
                tab1, tab2, tab3, tab4 = st.tabs(['📁 Sources', '📈 Metrics', '⚠️ Contradictions', '🔍 Debug'])

                with tab1:
                    # Source manifest
                    manifest = message['sources'].get('source_manifest', {})
                    if manifest.get('files'):
                        st.markdown('**Source Files:**')
                        for file in manifest.get('files', []):
                            details = manifest.get('file_details', {}).get(file, {})
                            triple_count = details.get('used_in_triples', 0)
                            chunk_count = details.get('used_in_chunks', 0)

                            st.markdown(
                                f"- `{file}` — "
                                f"Triples: {triple_count}, Chunks: {chunk_count}"
                            )

                    st.divider()

                    # Retrieved knowledge
                    triples = message['sources'].get('triples', [])
                    if triples:
                        st.markdown('**Retrieved Knowledge:**')
                        for i, triple in enumerate(triples[:5], 1):
                            st.markdown(
                                f"{i}. {triple['natural_text']}  \n"
                                f"   *Source: {triple.get('source_file', 'unknown')} • "
                                f"Confidence: {triple['confidence']:.2f} • "
                                f"Similarity: {triple.get('similarity_score', 0):.3f}*"
                            )

                    # Retrieved chunks
                    chunks = message['sources'].get('chunks', [])
                    if chunks:
                        st.divider()
                        st.markdown('**Document Excerpts:**')
                        for i, chunk in enumerate(chunks[:3], 1):
                            preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                            st.markdown(
                                f"**{i}. {chunk['doc_filename']}** (chunk {chunk.get('chunk_index', 0)})  \n"
                                f"*Similarity: {chunk.get('similarity_score', 0):.3f}*"
                            )
                            st.markdown(f"> {preview}")

                with tab2:
                    # Key metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric('Total Latency', f"{metrics.get('total_latency_ms', 0):.0f}ms")
                    col2.metric('Sources Retrieved',
                               message['stats']['triples_retrieved'] + message['stats']['chunks_retrieved'])
                    col3.metric('Confidence Score', f"{confidence:.0%}")

                    st.divider()

                    # Retrieval quality
                    retrieval_quality = metrics.get('retrieval_quality', {})
                    if retrieval_quality:
                        st.markdown('**Retrieval Quality:**')
                        col1, col2, col3 = st.columns(3)
                        col1.metric('Triples Found', retrieval_quality.get('triples_found', 0))
                        col2.metric('Chunks Found', retrieval_quality.get('chunks_found', 0))
                        col3.metric('Fallback Used',
                                   '✓' if retrieval_quality.get('fallback_triggered') else '✗')

                    st.divider()

                    # Latency breakdown chart
                    if 'latency_breakdown' in metrics:
                        st.markdown('**Processing Time Breakdown:**')
                        breakdown = metrics['latency_breakdown']

                        # Clean up step names for display
                        clean_names = {
                            'triple_retrieval_ms': 'Triple Retrieval',
                            'chunk_retrieval_ms': 'Chunk Retrieval',
                            'fallback_search_ms': 'Fallback Search',
                            'graph_context_ms': 'Graph Context',
                            'contradiction_detection_ms': 'Contradiction Check',
                            'prompt_building_ms': 'Prompt Assembly',
                            'llm_generation_ms': 'LLM Generation'
                        }

                        display_breakdown = {
                            clean_names.get(k, k): v
                            for k, v in breakdown.items()
                        }

                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(display_breakdown.values()),
                                y=list(display_breakdown.keys()),
                                orientation='h',
                                marker_color='steelblue',
                                text=[f'{v:.0f}ms' for v in display_breakdown.values()],
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            height=300,
                            margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_title='Time (ms)',
                            yaxis_title='',
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    if contradictions:
                        st.markdown('**Detected Contradictions:**')
                        for i, contra in enumerate(contradictions, 1):
                            severity = contra.get('severity', 'medium')
                            severity_color = {
                                'high': '🔴',
                                'medium': '🟡',
                                'low': '🟢'
                            }.get(severity, '⚪')

                            st.markdown(f"**{severity_color} {i}. {contra.get('reason', 'Unknown reason')}**")
                            st.markdown(f"- **Source:** `{contra.get('source', 'unknown')}`")
                            st.markdown(f"- **Confidence:** {contra.get('confidence', 0):.2f}")
                            st.markdown(f"- **Type:** {contra.get('type', 'unknown')}")

                            if 'text' in contra:
                                preview = contra['text'][:150] + "..." if len(contra['text']) > 150 else contra['text']
                                st.markdown(f"> {preview}")

                            if i < len(contradictions):
                                st.divider()
                    else:
                        st.info('✓ No contradictions detected in retrieved sources', icon='✅')

                with tab4:
                    # Reasoning trace
                    if 'reasoning_trace' in metrics and hasattr(st.session_state, 'show_reasoning') and st.session_state.show_reasoning:
                        st.markdown('**Reasoning Trace:**')
                        for step in metrics['reasoning_trace']:
                            st.markdown(f"- **{step['step']}**: {step['result']}")

                        st.divider()

                    # Retrieval quality (detailed)
                    st.markdown('**Retrieval Quality Details:**')
                    st.json(metrics.get('retrieval_quality', {}))

                    st.divider()

                    # Confidence breakdown
                    if 'confidence_breakdown' in metrics:
                        st.markdown('**Confidence Calculation:**')
                        st.json(metrics['confidence_breakdown'])

# ============================================================================
# CHAT INPUT (Bottom-Pinned)
# ============================================================================
if prompt := st.chat_input('Ask a question about employees, projects, or policies...'):
    # Add user message
    st.session_state.chat_history.append({
        'role': 'user',
        'content': prompt,
        'timestamp': datetime.now().isoformat()
    })

    # Show user message immediately
    with st.chat_message('user'):
        st.markdown(prompt)

    # Generate response
    with st.chat_message('assistant'):
        # Show progressive loading states
        with st.status('Processing query...', expanded=False) as status:
            st.write('🔍 Searching knowledge graph...')

            try:
                # Query the pipeline
                response = st.session_state.rag_pipeline.query(prompt, verbose=False)

                st.write('💭 Analyzing sources...')
                st.write('✍️ Generating answer...')

                status.update(label='Complete!', state='complete', expanded=False)

            except Exception as e:
                status.update(label='Error', state='error')
                st.error(f'Failed to process query: {e}')
                st.session_state.session_stats['failed_queries'] += 1
                st.stop()

        # Display answer
        st.markdown(response['answer'])

        # Show confidence badge
        confidence = response.get('confidence', 0)
        if confidence >= 0.8:
            conf_badge = '🟢 High'
            conf_class = 'confidence-high'
        elif confidence >= 0.5:
            conf_badge = '🟡 Medium'
            conf_class = 'confidence-medium'
        else:
            conf_badge = '🔴 Low'
            conf_class = 'confidence-low'

        st.markdown(
            f'<span class="confidence-badge {conf_class}">'
            f'{conf_badge} Confidence ({confidence:.0%})'
            f'</span>',
            unsafe_allow_html=True
        )

        # Low confidence warning
        if confidence < 0.5:
            st.warning(
                'This answer has low confidence. Please verify details or try rephrasing your question.',
                icon='⚠️'
            )

        # Store full response in history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response['answer'],
            'timestamp': datetime.now().isoformat(),
            'metrics': response.get('metrics', {}),
            'stats': response.get('stats', {}),
            'sources': response.get('sources', {}),
            'confidence': confidence
        })

        # Update session stats
        st.session_state.session_stats['total_queries'] += 1

        if response.get('status') == 'success':
            st.session_state.session_stats['successful_queries'] += 1
        else:
            st.session_state.session_stats['failed_queries'] += 1

        st.session_state.session_stats['contradictions_found'] += response['stats'].get('contradictions_found', 0)

        # Update average latency
        total_queries = st.session_state.session_stats['total_queries']
        current_avg = st.session_state.session_stats['avg_latency_ms']
        new_latency = response['metrics'].get('total_latency_ms', 0)

        st.session_state.session_stats['avg_latency_ms'] = (
            (current_avg * (total_queries - 1) + new_latency) / total_queries
        )

        # Rerun to update UI
        st.rerun()