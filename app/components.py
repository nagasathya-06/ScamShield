"""
UI Components, Custom Styling, and Visual Badges for Scam Shield Streamlit App.
"""

from typing import List, Tuple, Dict, Any
import html
import re


def get_custom_css() -> str:
    """
    Returns custom CSS styles for the Scam Shield dashboard.
    """
    return """
    <style>
    /* Global styles */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .verdict-box {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 700;
    }
    .verdict-scam {
        background-color: #fde8e8;
        border: 2px solid #e02424;
        color: #9b1c1c;
    }
    .verdict-safe {
        background-color: #def7ec;
        border: 2px solid #0e9f6e;
        color: #03543f;
    }
    .threat-card {
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        border-left: 5px solid;
        background-color: #f8fafc;
    }
    .token-scam {
        background-color: #ffcccc;
        color: #900;
        padding: 2px 5px;
        border-radius: 4px;
        font-weight: 600;
    }
    .token-safe {
        background-color: #d4edda;
        color: #155724;
        padding: 2px 5px;
        border-radius: 4px;
        font-weight: 600;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        margin-right: 5px;
    }
    .badge-critical { background-color: #f8d7da; color: #721c24; }
    .badge-high { background-color: #fff3cd; color: #856404; }
    .badge-medium { background-color: #e2e3e5; color: #383d41; }
    </style>
    """


def render_verdict_card(is_scam: bool, confidence: float, model_name: str) -> str:
    """
    Renders an HTML card displaying the classification verdict and confidence.
    """
    if is_scam:
        risk_level = "CRITICAL RISK" if confidence > 0.85 else "HIGH RISK"
        return f"""
        <div class="verdict-box verdict-scam">
            <h2 style="margin:0; font-size:1.8rem; color:#9b1c1c;">🚨 SCAM / FRAUDULENT MESSAGE</h2>
            <p style="margin:6px 0 0 0; font-size:1.1rem; color:#9b1c1c;">
                Classification: <strong>SCAM</strong> &bull; Threat Level: <strong>{risk_level}</strong>
            </p>
            <p style="margin:4px 0 0 0; font-size:0.95rem; color:#c81e1e;">
                Model: <strong>{model_name}</strong> | Confidence: <strong>{confidence*100:.2f}%</strong>
            </p>
        </div>
        """
    else:
        return f"""
        <div class="verdict-box verdict-safe">
            <h2 style="margin:0; font-size:1.8rem; color:#03543f;">✅ SAFE / LEGITIMATE MESSAGE</h2>
            <p style="margin:6px 0 0 0; font-size:1.1rem; color:#03543f;">
                Classification: <strong>SAFE</strong> &bull; Threat Level: <strong>MINIMAL / CLEAN</strong>
            </p>
            <p style="margin:4px 0 0 0; font-size:0.95rem; color:#046c4e;">
                Model: <strong>{model_name}</strong> | Legitimacy Confidence: <strong>{(1-confidence)*100:.2f}%</strong>
            </p>
        </div>
        """


def render_threat_item(threat: Dict[str, Any]) -> str:
    """
    Renders a single heuristic threat indicator card.
    """
    border_color = threat.get("color", "#e74c3c")
    severity = threat.get("severity", "HIGH")
    badge_cls = "badge-critical" if severity == "CRITICAL" else ("badge-high" if severity == "HIGH" else "badge-medium")

    return f"""
    <div class="threat-card" style="border-left-color: {border_color};">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong style="color: {border_color}; font-size: 1rem;">{html.escape(threat['title'])}</strong>
            <span class="metric-badge {badge_cls}">{severity}</span>
        </div>
        <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #475569;">
            {html.escape(threat['description'])}
        </p>
    </div>
    """


def render_highlighted_text(text: str, scam_words: List[str], safe_words: List[str]) -> str:
    """
    Highlights scam words in red and safe words in green inside the message text.
    """
    words = text.split()
    highlighted_tokens = []

    scam_set = set(w.lower() for w in scam_words)
    safe_set = set(w.lower() for w in safe_words)

    for word in words:
        clean_w = re.sub(r'[^a-zA-Z0-9]', '', word).lower()
        if clean_w in scam_set:
            highlighted_tokens.append(f'<span class="token-scam">{html.escape(word)}</span>')
        elif clean_w in safe_set:
            highlighted_tokens.append(f'<span class="token-safe">{html.escape(word)}</span>')
        else:
            highlighted_tokens.append(html.escape(word))

    return " ".join(highlighted_tokens)
