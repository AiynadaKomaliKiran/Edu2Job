import ast
import io
import os
import re
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
	import numpy as np
	from PIL import Image
except Exception:
	np = None
	Image = None

try:
	import fitz
except Exception:
	fitz = None

try:
	from rapidocr_onnxruntime import RapidOCR
except Exception:
	RapidOCR = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_FILE = os.path.join(BASE_DIR, "jobs_data.csv")
USERS_FILE = os.path.join(BASE_DIR, "users_data.csv")
HISTORY_FILE = os.path.join(BASE_DIR, "history_data.csv")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback_data.csv")

DOMAINS = ["IT", "Business", "Creative", "Core"]

UI_THEME = {
	"bg": "#0d1020",
	"bg_alt": "#151a33",
	"surface": "rgba(19, 23, 42, 0.84)",
	"surface_soft": "rgba(255, 255, 255, 0.04)",
	"border": "rgba(255, 255, 255, 0.10)",
	"text": "#f5f7ff",
	"muted": "#a8afcc",
	"accent": "#a855f7",
	"accent_2": "#7c3aed",
	"accent_soft": "rgba(168, 85, 247, 0.18)",
	"success": "#37d67a",
	"warning": "#ffb547",
	"danger": "#ff6b6b",
}

COMMON_SKILLS = {
	"python",
	"java",
	"c++",
	"sql",
	"excel",
	"power bi",
	"tableau",
	"machine learning",
	"deep learning",
	"data analysis",
	"data visualization",
	"statistics",
	"pandas",
	"numpy",
	"scikit-learn",
	"tensorflow",
	"pytorch",
	"html",
	"css",
	"javascript",
	"react",
	"nodejs",
	"communication",
	"leadership",
	"problem solving",
	"figma",
	"photoshop",
	"illustrator",
	"autocad",
	"cad",
	"solidworks",
	"thermodynamics",
	"structural engineering",
	"digital marketing",
	"seo",
	"content writing",
	"video editing",
	"ai",
	"nlp",
	"cloud",
	"aws",
	"docker",
	"kubernetes",
}

COURSE_CATALOG = {
	"python": "Python for Everybody",
	"machine learning": "Machine Learning Specialization",
	"deep learning": "Deep Learning Specialization",
	"feature engineering": "Feature Engineering for Machine Learning",
	"model deployment": "Machine Learning Engineering for Production (MLOps)",
	"mlops": "MLOps Fundamentals",
	"data science": "Data Science Professional Certificate",
	"data engineering": "Data Engineering with Python",
	"sql": "SQL for Data Analysis",
	"excel": "Excel Skills for Business",
	"power bi": "Power BI Data Analyst",
	"tableau": "Tableau Desktop Specialist",
	"statistics": "Statistics for Data Science",
	"data analysis": "Data Analytics Professional Certificate",
	"javascript": "Modern JavaScript Bootcamp",
	"react": "React Complete Guide",
	"nodejs": "Node.js Backend Development",
	"html": "Web Design Basics",
	"css": "Advanced CSS and Sass",
	"figma": "Figma UI UX Essentials",
	"photoshop": "Adobe Photoshop Masterclass",
	"illustrator": "Adobe Illustrator for Design",
	"autocad": "AutoCAD Complete Course",
	"cad": "CAD Design Fundamentals",
	"solidworks": "SolidWorks Beginner to Pro",
	"digital marketing": "Digital Marketing Strategy",
	"seo": "SEO Foundations",
	"content writing": "Content Writing Mastery",
	"communication": "Business Communication Skills",
	"leadership": "Leadership and Management",
	"aws": "AWS Cloud Practitioner",
	"docker": "Docker and Kubernetes Fundamentals",
}

ROLE_COURSE_PACKS = {
	"data scientist": ["python", "statistics", "machine learning", "deep learning", "feature engineering", "model deployment"],
	"data analyst": ["excel", "sql", "power bi", "tableau", "data analysis", "statistics"],
	"ml engineer": ["python", "machine learning", "deep learning", "mlops", "model deployment", "docker", "aws"],
	"ai engineer": ["python", "machine learning", "deep learning", "model deployment", "mlops", "docker"],
	"software engineer": ["python", "javascript", "react", "nodejs", "sql", "communication"],
	"web developer": ["html", "css", "javascript", "react", "nodejs"],
	"cloud engineer": ["aws", "docker", "kubernetes", "python"],
}

DOMAIN_COURSE_PACKS = {
	"it": ["python", "sql", "machine learning", "data analysis", "aws", "docker", "communication"],
	"business": ["excel", "power bi", "tableau", "communication", "leadership", "data analysis"],
	"creative": ["figma", "photoshop", "illustrator", "content writing", "communication"],
	"core": ["autocad", "cad", "solidworks", "statistics", "communication"],
}

SKILL_ALIASES = {
	"ml": "machine learning",
	"ai/ml": "machine learning",
	"artificial intelligence": "ai",
	"data analytics": "data analysis",
	"scikit learn": "scikit-learn",
	"sklearn": "scikit-learn",
	"nlp": "natural language processing",
	"natural-language-processing": "natural language processing",
}

EXTRA_SKILLS = {
	"feature engineering",
	"model deployment",
	"mlops",
	"data science",
	"data engineering",
	"natural language processing",
}

SKILL_VOCABULARY = set(COMMON_SKILLS) | set(COURSE_CATALOG.keys()) | EXTRA_SKILLS | set(SKILL_ALIASES.keys())

RELATED_ROLE_HINTS = {
	"video editing": ["Content Creator", "Video Editor", "Graphic Designer", "UI UX Designer", "Journalist"],
	"content writing": ["Content Creator", "Journalist", "Marketing Manager", "Teacher"],
	"figma": ["UI UX Designer", "Graphic Designer", "Web Developer"],
	"photoshop": ["Graphic Designer", "Content Creator", "UI UX Designer"],
	"illustrator": ["Graphic Designer", "Content Creator", "UI UX Designer"],
	"python": ["Data Scientist", "Machine Learning Engineer", "Software Engineer", "DevOps Engineer"],
	"sql": ["Data Analyst", "Business Analyst", "Financial Analyst"],
	"excel": ["Data Analyst", "Business Analyst", "Financial Analyst"],
	"communication": ["Business Analyst", "Sales Manager", "Teacher", "Journalist", "HR Manager"],
	"machine learning": ["Data Scientist", "Machine Learning Engineer", "AI Engineer", "DevOps Engineer"],
	"digital marketing": ["Marketing Manager", "Content Creator", "Sales Manager"],
}


def inject_ui_styles() -> None:
	css_str = f"""
	<style>
		:root {{
			--bg: {UI_THEME['bg']};
			--bg-alt: {UI_THEME['bg_alt']};
			--surface: {UI_THEME['surface']};
			--surface-soft: {UI_THEME['surface_soft']};
			--border: {UI_THEME['border']};
			--text: {UI_THEME['text']};
			--muted: {UI_THEME['muted']};
			--accent: {UI_THEME['accent']};
			--accent-2: {UI_THEME['accent_2']};
			--accent-soft: {UI_THEME['accent_soft']};
		}}

		.stApp {{
			background:
				radial-gradient(circle at top left, rgba(168, 85, 247, 0.14), transparent 32%),
				radial-gradient(circle at top right, rgba(124, 58, 237, 0.12), transparent 30%),
				linear-gradient(180deg, #0a0d1a 0%, var(--bg) 42%, #10152a 100%);
			color: var(--text);
			font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
		}}

		.stApp::before {{
			content: "";
			position: fixed;
			inset: 0;
			pointer-events: none;
			background-image: linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
			background-size: 64px 64px;
			opacity: 0.22;
			mask-image: linear-gradient(to bottom, rgba(0,0,0,0.7), transparent 88%);
		}}

		[data-testid="stSidebar"] {{
			background: linear-gradient(180deg, rgba(13, 16, 32, 0.98), rgba(17, 21, 42, 0.94));
			border-right: 1px solid var(--border);
		}}

		[data-testid="stSidebar"] > div {{
			padding-top: 1.4rem;
		}}

		[data-testid="stSidebar"] * {{
			color: var(--text);
		}}

		[data-testid="stSidebarNav"] {{
			display: none;
		}}

		h1, h2, h3, h4, h5, h6, p, label, span, div {{
			color: var(--text);
		}}

		.block-container {{
			padding-top: 1.2rem;
			padding-bottom: 2.5rem;
			max-width: 1280px;
		}}

		.page-shell {{
			background: linear-gradient(180deg, rgba(17, 21, 42, 0.72), rgba(13, 16, 32, 0.42));
			border: 1px solid var(--border);
			box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
			backdrop-filter: blur(18px);
			-webkit-backdrop-filter: blur(18px);
			border-radius: 28px;
			padding: 1.25rem 1.25rem 1.5rem;
			margin-bottom: 1rem;
		}}

		.hero-banner {{
			position: relative;
			overflow: hidden;
			background:
				linear-gradient(135deg, rgba(168, 85, 247, 0.22), rgba(124, 58, 237, 0.10)),
				linear-gradient(180deg, rgba(17, 21, 42, 0.98), rgba(12, 15, 28, 0.82));
			border: 1px solid rgba(255, 255, 255, 0.11);
			border-radius: 28px;
			padding: 1.5rem 1.7rem;
			box-shadow: 0 18px 60px rgba(0, 0, 0, 0.26);
			margin-bottom: 1rem;
		}}

		.hero-banner::after {{
			content: "";
			position: absolute;
			inset: 0;
			background: radial-gradient(circle at top right, rgba(255,255,255,0.09), transparent 30%);
			pointer-events: none;
		}}

		.hero-kicker {{
			display: inline-flex;
			align-items: center;
			gap: 0.5rem;
			padding: 0.35rem 0.7rem;
			border-radius: 999px;
			background: rgba(255, 255, 255, 0.06);
			border: 1px solid rgba(255, 255, 255, 0.1);
			color: #ddd9ff;
			font-size: 0.82rem;
			letter-spacing: 0.06em;
			text-transform: uppercase;
			margin-bottom: 0.9rem;
		}}

		.hero-title {{
			font-size: clamp(2rem, 4vw, 3.6rem);
			line-height: 1.05;
			font-weight: 800;
			letter-spacing: -0.03em;
			margin: 0 0 0.55rem;
		}}

		.hero-subtitle {{
			max-width: 760px;
			color: var(--muted);
			font-size: 1rem;
			line-height: 1.6;
			margin-bottom: 1.15rem;
		}}

		.page-card, .soft-card {{
			background: var(--surface);
			border: 1px solid var(--border);
			border-radius: 22px;
			box-shadow: 0 14px 40px rgba(0, 0, 0, 0.16);
		}}

		.page-card {{
			padding: 1.15rem 1.15rem 1.2rem;
		}}

		.soft-card {{
			padding: 1rem 1.05rem;
			background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
		}}

		.metric-card {{
			padding: 1rem 1.1rem;
			border-radius: 20px;
			background: linear-gradient(135deg, rgba(168,85,247,0.16), rgba(255,255,255,0.04));
			border: 1px solid rgba(168, 85, 247, 0.18);
		}}

		.metric-label {{
			font-size: 0.85rem;
			color: var(--muted);
			text-transform: uppercase;
			letter-spacing: 0.08em;
		}}

		.metric-value {{
			font-size: 1.65rem;
			font-weight: 800;
			margin-top: 0.2rem;
		}}

		.metric-desc {{
			margin-top: 0.25rem;
			color: var(--muted);
			font-size: 0.92rem;
		}}

		.stButton > button {{
			background: linear-gradient(135deg, var(--accent), var(--accent-2));
			color: white;
			border: 0;
			border-radius: 14px;
			padding: 0.72rem 1rem;
			font-weight: 700;
			box-shadow: 0 12px 24px rgba(124, 58, 237, 0.28);
			transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
		}}

		.stButton > button:hover {{
			transform: translateY(-1px);
			box-shadow: 0 18px 32px rgba(124, 58, 237, 0.36);
			filter: brightness(1.05);
		}}

		.stButton > button:active {{
			transform: translateY(0);
		}}

		/* Highlight the primary form submit action (Get Recommendations) */
		[data-testid="stFormSubmitButton"] button {{
			background: linear-gradient(135deg, #c4b5fd, #a78bfa) !important;
			color: #ffffff !important;
			border: 1px solid rgba(255, 255, 255, 0.25) !important;
			border-radius: 14px !important;
			padding: 0.8rem 1rem !important;
			box-shadow: 0 12px 24px rgba(167, 139, 250, 0.32) !important;
		}}

		[data-testid="stFormSubmitButton"] button:hover {{
			filter: brightness(1.06) !important;
		}}

		[data-testid="stFileUploaderDropzone"] {{
			background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025));
			border: 1px dashed rgba(168,85,247,0.45);
			border-radius: 18px;
			color: var(--text) !important;
		}}

		[data-testid="stFileUploaderDropzone"] p,
		[data-testid="stFileUploaderDropzone"] span,
		[data-testid="stFileUploaderDropzone"] label {{
			color: var(--text) !important;
		}}

		[data-testid="stFileUploaderDropzone"] button {{
			background: #f8fafc !important;
			color: #0f172a !important;
			border: 1px solid rgba(15, 23, 42, 0.18) !important;
			border-radius: 12px !important;
			box-shadow: none !important;
		}}

		[data-testid="stFileUploaderDropzone"] button:hover {{
			background: #ffffff !important;
			color: #020617 !important;
		}}

		[data-testid="stFileUploaderDropzone"] button * {{
			color: #0f172a !important;
		}}

		[data-testid="stTextInput"] input,
		[data-testid="stTextArea"] textarea,
		[data-testid="stNumberInput"] input {{
			background: #f8fafc !important;
			border-color: rgba(15, 23, 42, 0.18) !important;
			color: #4b5563 !important;
			-webkit-text-fill-color: #4b5563 !important;
			caret-color: #4b5563 !important;
			border-radius: 14px !important;
		}}

		[data-baseweb="select"] > div,
		[data-baseweb="select"] input {{
			background: linear-gradient(135deg, rgba(167, 139, 250, 0.45), rgba(139, 92, 246, 0.40)) !important;
			border-color: rgba(221, 214, 254, 0.9) !important;
			color: #ffffff !important;
			-webkit-text-fill-color: #ffffff !important;
			caret-color: #ffffff !important;
			border-radius: 14px !important;
		}}

		[data-baseweb="select"] > div:hover,
		[data-baseweb="select"] > div:focus-within {{
			border-color: rgba(221, 214, 254, 0.9) !important;
			box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.22) !important;
		}}

		[data-baseweb="select"] svg {{
			fill: #ffffff !important;
		}}

		[data-testid="stTextInput"] input::placeholder,
		[data-testid="stTextArea"] textarea::placeholder,
		[data-testid="stNumberInput"] input::placeholder {{
			color: rgba(75, 85, 99, 0.62) !important;
		}}

		[data-testid="stTextInput"] input:focus,
		[data-testid="stTextArea"] textarea:focus,
		[data-testid="stNumberInput"] input:focus {{
			box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.18) !important;
		}}

		[data-testid="stTextInput"] input:-webkit-autofill,
		[data-testid="stTextArea"] textarea:-webkit-autofill,
		[data-testid="stNumberInput"] input:-webkit-autofill {{
			-webkit-text-fill-color: #4b5563 !important;
			box-shadow: 0 0 0 1000px #f8fafc inset !important;
		}}

		[data-baseweb="popover"],
		[data-baseweb="menu"],
		[role="listbox"] {{
			background: linear-gradient(180deg, rgba(180, 146, 255, 0.98), rgba(155, 115, 246, 0.98)) !important;
			border: 1px solid rgba(237, 233, 254, 0.55) !important;
			box-shadow: 0 22px 50px rgba(109, 40, 217, 0.28) !important;
			border-radius: 16px !important;
		}}

		[data-baseweb="popover"] [role="option"],
		[data-baseweb="menu"] [role="option"],
		[data-baseweb="select"] [role="option"],
		[role="listbox"] [role="option"],
		[role="option"] {{
			background: linear-gradient(135deg, rgba(178, 140, 255, 0.95), rgba(157, 118, 247, 0.95)) !important;
			color: #ffffff !important;
			padding: 0.7rem 0.9rem !important;
			font-weight: 600 !important;
			border-radius: 10px !important;
			margin: 0.2rem 0.35rem !important;
		}}

		[data-baseweb="popover"] [role="option"]:hover,
		[data-baseweb="popover"] [role="option"][aria-selected="true"],
		[data-baseweb="menu"] [role="option"]:hover,
		[data-baseweb="menu"] [role="option"][aria-selected="true"],
		[data-baseweb="select"] [role="option"]:hover,
		[data-baseweb="select"] [role="option"][aria-selected="true"],
		[role="listbox"] [role="option"]:hover,
		[role="listbox"] [role="option"][aria-selected="true"],
		[role="option"]:hover,
		[role="option"][aria-selected="true"] {{
			background: linear-gradient(135deg, rgba(124, 58, 237, 0.92), rgba(168, 85, 247, 0.88)) !important;
			color: #ffffff !important;
		}}

		[data-baseweb="select"] input::placeholder {{
			color: rgba(255, 255, 255, 0.88) !important;
		}}

		[data-testid="stTabs"] button {{
			color: var(--muted) !important;
			font-weight: 700;
		}}

		[data-testid="stTabs"] button[aria-selected="true"] {{
			color: var(--text) !important;
		}}

		.stDataFrame {{
			border: 1px solid rgba(255,255,255,0.08);
			border-radius: 18px;
			overflow: hidden;
		}}

		.chart-wrap {{
			padding: 0.45rem 0.35rem 0.6rem;
			margin: 0.25rem auto 0.9rem;
			overflow: hidden;
			text-align: center;
		}}

		.chart-wrap [data-testid="stImage"] img,
		.chart-wrap canvas {{
			max-width: 720px !important;
			margin: 0 auto !important;
		}}

		[data-testid="stToolbar"],
		[data-testid="stElementToolbar"] {{
			opacity: 1 !important;
			visibility: visible !important;
			display: flex !important;
		}}

		[data-testid="stDataFrame"] button,
		[data-testid="stDataFrame"] [role="button"],
		[data-testid="stDataFrame"] [aria-haspopup="menu"] {{
			background: linear-gradient(135deg, #c4b5fd, #a78bfa) !important;
			color: #ffffff !important;
			border-radius: 8px !important;
			border: none !important;
			box-shadow: 0 4px 10px rgba(167, 139, 250, 0.4) !important;
			opacity: 1 !important;
		}}

		[data-testid="stDataFrame"] button svg,
		[data-testid="stDataFrame"] [role="button"] svg,
		[data-testid="stDataFrame"] [aria-haspopup="menu"] svg {{
			fill: #ffffff !important;
			stroke: #ffffff !important;
		}}

		[data-testid="stDataFrame"] button:hover,
		[data-testid="stDataFrame"] [role="button"]:hover,
		[data-testid="stDataFrame"] [aria-haspopup="menu"]:hover {{
			filter: brightness(1.1) !important;
		}}

		[data-testid="stDataFrame"] [data-baseweb="popover"],
		[data-testid="stDataFrame"] [data-baseweb="menu"],
		[data-testid="stDataFrame"] [role="listbox"],
		[data-testid="stDataFrame"] [role="menu"] {{
			background: linear-gradient(180deg, rgba(196, 181, 253, 0.98), rgba(167, 139, 250, 0.98)) !important;
			border: 1px solid rgba(237, 233, 254, 0.65) !important;
			box-shadow: 0 22px 50px rgba(109, 40, 217, 0.32) !important;
			border-radius: 14px !important;
		}}

		[data-testid="stDataFrame"] [data-baseweb="popover"] *,
		[data-testid="stDataFrame"] [data-baseweb="menu"] *,
		[data-testid="stDataFrame"] [role="listbox"] *,
		[data-testid="stDataFrame"] [role="menu"] * {{
			color: #ffffff !important;
		}}

		[data-testid="stDataFrame"] [role="option"],
		[data-testid="stDataFrame"] [role="menuitem"] {{
			background: transparent !important;
			color: #ffffff !important;
		}}

		[data-testid="stDataFrame"] [role="option"]:hover,
		[data-testid="stDataFrame"] [role="menuitem"]:hover,
		[data-testid="stDataFrame"] [aria-selected="true"] {{
			background: linear-gradient(135deg, rgba(124, 58, 237, 0.92), rgba(168, 85, 247, 0.88)) !important;
			color: #ffffff !important;
		}}

		[data-testid="stElementToolbar"] button,
		[data-testid="stToolbar"] button,
		[data-testid="StyledFullScreenButton"],
		[data-testid="StyledFullScreenButton"] button {{
			background: linear-gradient(135deg, #c4b5fd, #a78bfa) !important;
			color: #ffffff !important;
			border-radius: 8px !important;
			border: none !important;
			box-shadow: 0 4px 10px rgba(167, 139, 250, 0.4) !important;
			opacity: 1 !important;
		}}

		button[title="View fullscreen"],
		button[aria-label="View fullscreen"] {{
			background: linear-gradient(135deg, #c4b5fd, #a78bfa) !important;
			color: #ffffff !important;
			border-radius: 8px !important;
			border: none !important;
			box-shadow: 0 4px 10px rgba(167, 139, 250, 0.4) !important;
			opacity: 1 !important;
		}}

		button[title="View fullscreen"]:hover,
		button[aria-label="View fullscreen"]:hover,
		[data-testid="stElementToolbar"] button:hover,
		[data-testid="stToolbar"] button:hover {{
			filter: brightness(1.1) !important;
		}}

		button[title="View fullscreen"] svg,
		button[aria-label="View fullscreen"] svg,
		[data-testid="stElementToolbar"] button svg,
		[data-testid="stToolbar"] button svg {{
			fill: #ffffff !important;
			stroke: #ffffff !important;
		}}

		[data-testid="stMetricValue"] {{
			color: var(--text);
		}}

		[data-testid="stMetricDelta"] {{
			color: var(--muted);
		}}

		[data-testid="stAlert"] {{
			border-radius: 16px;
		}}

		@media (max-width: 768px) {{
			.block-container {{
				padding-left: 0.75rem;
				padding-right: 0.75rem;
			}}
			.hero-banner {{
				padding: 1.1rem;
			}}
			.page-shell {{
				border-radius: 22px;
				padding: 0.9rem;
			}}
		}}
	</style>
	"""
	st.markdown(css_str, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, kicker: str = "Career intelligence") -> None:
	st.markdown(
		f"""
		<div class="hero-banner">
			<div class="hero-kicker">{kicker}</div>
			<div class="hero-title" style="font-family: 'Times New Roman', Times, serif; font-style: italic;">{title}</div>
			<div class="hero-subtitle">{subtitle}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_section_title(title: str, description: str = "") -> None:
	if description:
		st.markdown(
			f"""
			<div class="page-card" style="margin-bottom: 0.9rem;">
				<div style="font-size: 1.25rem; font-weight: 800; margin-bottom: 0.25rem;">{title}</div>
				<div style="color: var(--muted); line-height: 1.6;">{description}</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
	else:
		st.markdown(
			f"""
			<div class="page-card" style="margin-bottom: 0.9rem;">
				<div style="font-size: 1.25rem; font-weight: 800;">{title}</div>
			</div>
			""",
			unsafe_allow_html=True,
		)


def render_metric_card(label: str, value: str, desc: str) -> None:
	st.markdown(
		f"""
		<div class="metric-card">
			<div class="metric-label">{label}</div>
			<div class="metric-value">{value}</div>
			<div class="metric-desc">{desc}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def init_files() -> None:
	if not os.path.exists(USERS_FILE):
		pd.DataFrame(columns=["username", "password", "name"]).to_csv(USERS_FILE, index=False)
	if not os.path.exists(HISTORY_FILE):
		pd.DataFrame(
			columns=[
				"timestamp",
				"username",
				"mode",
				"name",
				"input_text",
				"detected_skills",
				"domain_suggestion",
				"top_jobs",
				"top_scores",
			],
		).to_csv(HISTORY_FILE, index=False)
	if not os.path.exists(FEEDBACK_FILE):
		pd.DataFrame(columns=["timestamp", "username", "rating", "feedback"]).to_csv(FEEDBACK_FILE, index=False)


@st.cache_data
def load_jobs() -> pd.DataFrame:
	jobs_df = pd.read_csv(JOBS_FILE)
	jobs_df["Skills"] = jobs_df["Skills"].fillna("").astype(str)
	jobs_df["Domain"] = jobs_df["Domain"].fillna("Unknown").astype(str)
	jobs_df["Job Role"] = jobs_df["Job Role"].fillna("Unknown Role").astype(str)
	jobs_df["combined_text"] = jobs_df["Job Role"].str.lower() + " " + jobs_df["Skills"].str.lower() + " " + jobs_df["Domain"].str.lower()
	return jobs_df


def preprocess_text(text: str) -> str:
	tokens = [tok.strip() for tok in text.lower().replace("\n", " ").split()]
	cleaned = [tok for tok in tokens if tok.isalnum() and tok not in ENGLISH_STOP_WORDS]
	return " ".join(cleaned)


def extract_skills_from_text(text: str, skills_universe: set[str]) -> list[str]:
	t = text.lower()
	found = []
	for skill in skills_universe:
		if " " in skill:
			if skill in t:
				found.append(skill)
		else:
			parts = t.replace(",", " ").replace(";", " ").split()
			if skill in parts:
				found.append(skill)
	return sorted(set(found))


def extract_resume_text(uploaded_pdf) -> tuple[str, str]:
	if uploaded_pdf is None:
		return "", "No file was uploaded."
	try:
		pdf_bytes = uploaded_pdf.getvalue()
		reader = PdfReader(io.BytesIO(pdf_bytes))
		if reader.is_encrypted:
			try:
				reader.decrypt("")
			except Exception:
				return "", "The PDF is encrypted or password-protected."
		full_text = []
		for page in reader.pages:
			full_text.append(page.extract_text() or "")
		text = "\n".join(full_text).strip()
		if not text:
			return extract_resume_text_with_ocr(pdf_bytes)
		return text, ""
	except Exception as exc:
		return "", f"PDF parsing failed: {exc}"


@st.cache_resource
def get_ocr_engine():
	if RapidOCR is None:
		return None
	return RapidOCR()


def ocr_image_to_text(image) -> str:
	engine = get_ocr_engine()
	if engine is None or np is None:
		return ""
	ocr_result = engine(np.array(image))
	if not ocr_result:
		return ""
	lines = ocr_result[0] if isinstance(ocr_result, tuple) else ocr_result
	texts = []
	for item in lines or []:
		if isinstance(item, (list, tuple)) and len(item) >= 2:
			texts.append(str(item[1]))
		elif item:
			texts.append(str(item))
	return "\n".join(texts).strip()


def extract_ocr_from_embedded_images(pdf_bytes: bytes) -> str:
	if Image is None:
		return ""
	try:
		reader = PdfReader(io.BytesIO(pdf_bytes))
		chunks = []
		for page in reader.pages:
			for img in getattr(page, "images", []):
				try:
					image = Image.open(io.BytesIO(img.data)).convert("RGB")
					text = ocr_image_to_text(image)
					if text:
						chunks.append(text)
				except Exception:
					continue
		return "\n".join(chunks).strip()
	except Exception:
		return ""


def extract_resume_text_with_ocr(pdf_bytes: bytes) -> tuple[str, str]:
	if Image is None or RapidOCR is None or np is None:
		return "", (
			"Scanned PDF OCR dependencies are missing. Install: "
			"rapidocr-onnxruntime, Pillow, numpy"
		)

	if fitz is None:
		fallback_text = extract_ocr_from_embedded_images(pdf_bytes)
		if fallback_text:
			return fallback_text, ""
		return "", "OCR backend fitz is unavailable, and no embedded images were readable from this PDF."

	try:
		doc = fitz.open(stream=pdf_bytes, filetype="pdf")
		ocr_chunks = []
		for page in doc:
			pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
			image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
			page_text = ocr_image_to_text(image)
			if page_text:
				ocr_chunks.append(page_text)
		text = "\n".join(ocr_chunks).strip()
		if not text:
			return "", "The PDF appears to be scanned, but OCR could not detect readable text."
		return text, ""
	except Exception as exc:
		return "", f"OCR extraction failed: {exc}"


def recommend_jobs(profile_text: str, jobs_df: pd.DataFrame, domain_filter: str = "All", top_n: int = 5) -> pd.DataFrame:
	filtered_jobs = jobs_df.copy()
	if domain_filter in DOMAINS:
		filtered_jobs = filtered_jobs[filtered_jobs["Domain"].str.lower() == domain_filter.lower()].copy()
	if filtered_jobs.empty:
		return pd.DataFrame(columns=["Job Role", "Domain", "Skills", "score"])
	corpus = filtered_jobs["combined_text"].tolist() + [preprocess_text(profile_text)]
	vectorizer = TfidfVectorizer(ngram_range=(1, 2))
	tfidf = vectorizer.fit_transform(corpus)
	similarities = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
	max_similarity = float(similarities.max()) if len(similarities) > 0 else 0.0
	if max_similarity > 0:
		normalized_scores = (similarities / max_similarity) * 100
	else:
		normalized_scores = similarities
	results = filtered_jobs[["Job Role", "Domain", "Skills"]].copy().reset_index(drop=True)
	results["score"] = normalized_scores.round(2)
	results = results.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
	return results


def score_band(score: float) -> str:
	if score >= 40:
		return "Strong Fit"
	if 20 <= score < 40:
		return "Moderate Fit"
	return "Low Fit"


def suggest_domain(recommendations: pd.DataFrame) -> str:
	if recommendations.empty:
		return "Not enough data"
	domain_scores = recommendations.groupby("Domain", as_index=False)["score"].mean().sort_values("score", ascending=False)
	return str(domain_scores.iloc[0]["Domain"])


def normalize_skill(skill: str) -> str:
	cleaned = " ".join(skill.lower().replace("_", " ").replace("-", "-").split())
	return SKILL_ALIASES.get(cleaned, cleaned)


def extract_skill_phrases(text: str, vocabulary: set[str]) -> set[str]:
	normalized_text = re.sub(r"[;|]", ",", text.lower())
	normalized_text = normalized_text.replace("/", " ")
	normalized_text = re.sub(r"\s+", " ", normalized_text)
	found = set()
	# Longest-first keeps multi-word phrases like "machine learning" intact.
	for phrase in sorted(vocabulary, key=len, reverse=True):
		pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
		if re.search(pattern, normalized_text):
			found.add(normalize_skill(phrase))
	return found


def parse_job_skills(skills_string: str) -> set[str]:
	cleaned = skills_string.lower().replace("\n", " ").replace(";", ",")
	known = extract_skill_phrases(cleaned, SKILL_VOCABULARY)

	# Preserve explicit comma-separated phrases even if they are not in vocabulary.
	comma_phrases = set()
	for chunk in cleaned.split(","):
		part = " ".join(chunk.strip().split())
		if not part:
			continue
		# Keep concise phrases only (avoids one long sentence as a skill).
		if 1 <= len(part.split()) <= 4:
			comma_phrases.add(normalize_skill(part))

	return known | comma_phrases


def skill_gap_analysis(user_skills: list[str], top_job_skills: str) -> list[str]:
	user_set = {normalize_skill(s) for s in user_skills}
	# Also match skill phrases from the user side for better comparison (e.g., "ml" vs "machine learning").
	user_set |= extract_skill_phrases(", ".join(user_skills), SKILL_VOCABULARY)
	top_set = parse_job_skills(top_job_skills)
	missing = sorted(list(top_set - user_set))
	return missing[:10]


def career_path_message(top_job: str, missing_skills: list[str]) -> str:
	if not missing_skills:
		return f"You are well-aligned for {top_job}. Build projects and prepare interview case studies to secure the role."
	needed = ", ".join(missing_skills[:5])
	return f"You can become a {top_job} by learning {needed} and building practical projects around these skills."


def recommend_courses(missing_skills: list[str], top_job_role: str = "", top_domain: str = "") -> list[dict[str, str]]:
	selected: list[dict[str, str]] = []
	seen: set[str] = set()

	def build_item(course_name: str, trigger: str = "") -> dict[str, str]:
		trigger_text = trigger or "your current skill gap"
		course_lower = course_name.lower()
		if "portfolio" in course_lower or "projects" in course_lower:
			why = f"Helps you build proof-of-work around {trigger_text}."
			benefit = "Shows practical work experience in your resume."
		elif "github" in course_lower or "git" in course_lower:
			why = f"Strengthens workflow and collaboration skills around {trigger_text}."
			benefit = "Signals professionalism in project handling."
		elif "interview" in course_lower or "resume" in course_lower:
			why = "Improves your presentation and job-search readiness."
			benefit = "Makes your profile more interview-ready."
		elif "mlops" in course_lower or "deployment" in course_lower:
			why = f"Connects learning with production-ready delivery for {trigger_text}."
			benefit = "Adds deployment and production skills to your resume."
		elif "dashboard" in course_lower or "reporting" in course_lower or "power bi" in course_lower:
			why = f"Helps you translate {trigger_text} into business impact."
			benefit = "Shows you can deliver insights, not just analysis."
		elif "communication" in course_lower:
			why = "Improves how you explain projects, results, and ideas."
			benefit = "Boosts interview and stakeholder communication."
		else:
			why = f"Builds hands-on capability in {trigger_text}."
			benefit = "Makes your resume more practical and job-ready."
		return {"course": course_name, "why_it_helps": why, "resume_benefit": benefit}

	def add_item(course_name: str, trigger: str = "") -> None:
		clean_label = " ".join(course_name.split())
		if clean_label and clean_label not in seen:
			seen.add(clean_label)
			selected.append(build_item(clean_label, trigger))

	def add_skill_course(skill_key: str) -> None:
		course_name = COURSE_CATALOG.get(skill_key)
		if course_name:
			add_item(course_name, skill_key)

	normalized_missing = [normalize_skill(skill) for skill in missing_skills if skill.strip()]
	missing_tokens = {tok for phrase in normalized_missing for tok in phrase.split() if len(tok) > 2}
	role_key = top_job_role.lower().strip()
	domain_key = top_domain.lower().strip()

	related_token_map = {
		"deployment": ["model deployment", "mlops", "docker", "aws"],
		"feature": ["feature engineering", "machine learning", "data analysis"],
		"engineer": ["python", "sql", "docker", "aws"],
		"analysis": ["data analysis", "sql", "statistics", "excel"],
		"ai": ["machine learning", "deep learning", "python"],
		"model": ["machine learning", "deep learning", "model deployment"],
		"web": ["javascript", "react", "nodejs", "html", "css"],
		"cloud": ["aws", "docker", "kubernetes"],
	}

	# 1) Direct skill-to-course matches.
	for missing in normalized_missing:
		for skill_key in COURSE_CATALOG:
			if missing == skill_key or missing in skill_key or skill_key in missing:
				add_skill_course(skill_key)

	# 2) Related skill expansion.
	for token in sorted(missing_tokens):
		for related_key in related_token_map.get(token, []):
			add_skill_course(related_key)

	# 3) Role-specific resume boosters that look better on a profile than only skill-name repeats.
	role_boosters = {
		"data scientist": [
			"Applied Machine Learning Projects",
			"Feature Engineering for Real-World ML",
			"MLOps and Model Monitoring",
			"Python Portfolio Projects for Data Science",
			"Data Science Case Studies and Interview Prep",
		],
		"data analyst": [
			"Analytics Dashboard Projects with SQL and BI",
			"Excel for Business Intelligence",
			"Power BI Reporting Projects",
			"Data Storytelling and Presentation Skills",
			"Business Analysis Case Studies",
		],
		"ml engineer": [
			"Deploy ML Models with FastAPI and Docker",
			"MLOps and Model Monitoring",
			"Building ML Pipelines in Python",
			"Cloud Deployment for ML Systems",
			"Production-Ready Machine Learning Projects",
		],
		"ai engineer": [
			"Applied Deep Learning Projects",
			"Deploying AI Services with APIs",
			"MLOps for AI Products",
			"Python Projects for AI Engineers",
			"AI Portfolio and Interview Preparation",
		],
		"software engineer": [
			"Backend API Development with Node.js",
			"System Design Fundamentals",
			"Git, GitHub and Collaboration Workflows",
			"Testing and Debugging for Developers",
			"Portfolio Projects for Software Engineers",
		],
		"web developer": [
			"Modern Frontend Projects with React",
			"Responsive Web Design Projects",
			"JavaScript Problem Solving for Developers",
			"UI Engineering and Accessibility",
			"Portfolio Building for Web Developers",
		],
		"cloud engineer": [
			"AWS Cloud Architecture Projects",
			"Docker and Kubernetes for Deployment",
			"Infrastructure Basics for Engineers",
			"CI/CD Pipelines for Cloud Projects",
			"Cloud Portfolio Projects",
		],
	}

	for role_name, booster_courses in role_boosters.items():
		if role_name in role_key:
			for course_name in booster_courses:
				add_item(course_name, role_name)

	# 4) Domain-specific boosters so IT gets a more relevant mix than the generic list.
	domain_boosters = {
		"it": [
			"Git and GitHub for Developers",
			"REST API Development Projects",
			"SQL and Python Integration Projects",
			"System Design Basics for IT Roles",
			"Portfolio Building for Tech Jobs",
		],
		"business": [
			"Business Analytics Projects",
			"Executive Communication Skills",
			"Dashboard Reporting with Power BI",
			"Business Case Study Practice",
			"Interview Skills for Business Roles",
		],
		"creative": [
			"UI/UX Portfolio Projects",
			"Design Systems and Branding",
			"Creative Content Strategy",
			"Visual Communication Projects",
			"Client Presentation Skills",
		],
		"core": [
			"Engineering Project Documentation",
			"Technical Communication for Engineers",
			"CAD Project Portfolio",
			"Problem Solving for Core Roles",
			"Interview Preparation for Core Jobs",
		],
	}
	for course_name in domain_boosters.get(domain_key, []):
		add_item(course_name, domain_key)

	# 5) Final fallback: resume-friendly courses, not just the same skill list.
	fallback_courses = [
		"Git and GitHub for Developers",
		"Portfolio Building for Tech Jobs",
		"Interview Preparation and Resume Writing",
		"Project-Based Learning with Python",
		"Applied Problem Solving for Professionals",
		"Communication Skills for Interviews",
	]
	for course_name in fallback_courses:
		if len(selected) >= 5:
			break
		add_item(course_name, domain_key or role_key or "career growth")

	# Keep the output compact, useful, and varied.
	return selected[:5]


def suggest_related_roles(user_skills: list[str], jobs_df: pd.DataFrame, recommendations: pd.DataFrame | None = None, limit: int = 4) -> list[dict[str, str]]:
	role_lookup = {
		str(row["Job Role"]).strip().lower(): {"Job Role": str(row["Job Role"]), "Domain": str(row["Domain"])}
		for _, row in jobs_df[["Job Role", "Domain"]].drop_duplicates().iterrows()
	}
	role_scores: dict[str, dict[str, object]] = {}

	def add_role(role_name: str, score: float, reason: str) -> None:
		clean_role = " ".join(role_name.split())
		role_key = clean_role.lower()
		if role_key not in role_lookup:
			return
		existing = role_scores.get(role_key)
		payload = {
			"Job Role": role_lookup[role_key]["Job Role"],
			"Domain": role_lookup[role_key]["Domain"],
			"reason": reason,
			"score": score,
		}
		if existing is None or float(score) > float(existing["score"]):
			role_scores[role_key] = payload

	skill_set = {normalize_skill(skill) for skill in user_skills if skill.strip()}
	for skill in skill_set:
		for index, role_name in enumerate(RELATED_ROLE_HINTS.get(skill, [])):
			add_role(role_name, 100 - index * 5, f"Related to {skill}")

	if recommendations is not None and not recommendations.empty:
		for index, row in recommendations.head(limit + 2).iterrows():
			add_role(str(row["Job Role"]), 85 - index * 4, "Top recommendation")

		primary_domain = str(recommendations.iloc[0]["Domain"]).strip().lower()
		domain_roles = jobs_df[jobs_df["Domain"].str.lower() == primary_domain]
		for index, row in domain_roles.head(limit * 2).iterrows():
			add_role(str(row["Job Role"]), 60 - index * 2, f"Same domain: {row['Domain']}")

	ordered_roles = sorted(role_scores.values(), key=lambda item: (-float(item["score"]), str(item["Job Role"])))
	return [{"Job Role": str(item["Job Role"]), "Domain": str(item["Domain"]), "reason": str(item["reason"])} for item in ordered_roles[:limit]]


def append_history(username: str, mode: str, name: str, input_text: str, detected_skills: list[str], domain_suggestion: str, recommendations: pd.DataFrame) -> None:
	row = {
		"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"username": username,
		"mode": mode,
		"name": name,
		"input_text": input_text[:1200],
		"detected_skills": str(detected_skills),
		"domain_suggestion": domain_suggestion,
		"top_jobs": str(recommendations["Job Role"].tolist()),
		"top_scores": str(recommendations["score"].tolist()),
	}
	pd.DataFrame([row]).to_csv(HISTORY_FILE, mode="a", index=False, header=False)


def read_history(username: str) -> pd.DataFrame:
	df = pd.read_csv(HISTORY_FILE)
	if df.empty:
		return df
	return df[df["username"] == username].copy()


def save_feedback(username: str, rating: int, feedback_text: str) -> None:
	row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "username": username, "rating": rating, "feedback": feedback_text}
	pd.DataFrame([row]).to_csv(FEEDBACK_FILE, mode="a", index=False, header=False)


def ai_semantic_helper() -> str:
	api_key = os.getenv("CAREER_AI_API_KEY", "").strip()
	if not api_key:
		return ""
	return ""


def user_exists(username: str) -> bool:
	users = pd.read_csv(USERS_FILE)
	return username in users["username"].astype(str).tolist()


def create_user(username: str, password: str, name: str) -> tuple[bool, str]:
	if not username or not password or not name:
		return False, "Please fill all signup fields."
	if user_exists(username):
		return False, "Username already exists. Try another one."
	pd.DataFrame([{"username": username, "password": password, "name": name}]).to_csv(USERS_FILE, mode="a", index=False, header=False)
	return True, "Signup successful. Please login now."


def authenticate(username: str, password: str) -> tuple[bool, str]:
	users = pd.read_csv(USERS_FILE)
	match = users[(users["username"] == username) & (users["password"] == password)]
	if match.empty:
		return False, ""
	return True, str(match.iloc[0]["name"])


def auth_page() -> None:
	render_hero(
		"Edu2Job",
		"An AI-powered platform that analyzes your skills and provides personalized job recommendations, skill gap insights, and career guidance.",
		kicker="Career platform",
	)
	left, right = st.columns([1.05, 0.95], gap="large")
	with left:
		st.markdown(
			"""
			<div class="page-card" style="min-height: 100%;">
				<div style="font-size: 1.35rem; font-weight: 800; margin-bottom: 0.5rem;">Built for fast career guidance</div>
				<div style="color: var(--muted); line-height: 1.8;">
					Crafted to deliver quick, reliable, and intuitive career guidance with a seamless user experience.
				</div>
				<div style="display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.85rem; margin-top: 1rem;">
					<div class="soft-card"><div class="metric-label">Resume analyzer</div><div style="font-weight: 700; margin-top: 0.25rem;">Smart resume review</div></div>
					<div class="soft-card"><div class="metric-label">Manual input</div><div style="font-weight: 700; margin-top: 0.25rem;">Structured profile entry</div></div>
					<div class="soft-card"><div class="metric-label">Insights</div><div style="font-weight: 700; margin-top: 0.25rem;">Charts and history</div></div>
					<div class="soft-card"><div class="metric-label">Feedback</div><div style="font-weight: 700; margin-top: 0.25rem;">User rating system</div></div>
				</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
	with right:
		st.markdown('<div class="page-card">', unsafe_allow_html=True)
		tab_login, tab_signup = st.tabs(["Login", "Signup"])
		with tab_login:
			st.subheader("Login")
			l_user = st.text_input("Username", key="login_username")
			l_pass = st.text_input("Password", type="password", key="login_password")
			if st.button("Login", use_container_width=True):
				ok, name = authenticate(l_user.strip(), l_pass.strip())
				if ok:
					st.session_state.logged_in = True
					st.session_state.username = l_user.strip()
					st.session_state.name = name
					st.success("Login successful")
					st.rerun()
				else:
					st.error("Invalid credentials")
		with tab_signup:
			st.subheader("Create account")
			s_name = st.text_input("Full Name", key="signup_name")
			s_user = st.text_input("Choose Username", key="signup_username")
			s_pass = st.text_input("Choose Password", type="password", key="signup_password")
			if st.button("Signup", use_container_width=True):
				ok, msg = create_user(s_user.strip(), s_pass.strip(), s_name.strip())
				if ok:
					st.success(msg)
				else:
					st.warning(msg)
		st.markdown('</div>', unsafe_allow_html=True)


def page_dashboard() -> None:
	render_hero(
		"Dashboard",
		f"Welcome back, {st.session_state.name}. Here is a quick view of your skills and best career direction.",
		kicker="Overview",
	)
	detected = st.session_state.get("detected_skills", [])
	domain = st.session_state.get("recommended_domain", "Not available yet")
	detected_value = str(len(detected)) if detected else "No data available yet"
	detected_desc = "Skills identified from your latest profile." if detected else "Start by analyzing your resume."
	domain_value = domain if domain != "Not available yet" else "No data available yet"
	domain_desc = "Your best-fit career path." if domain != "Not available yet" else "Add your skills to see recommendations."
	c1, c2, c3 = st.columns(3, gap="medium")
	with c1:
		render_metric_card("Detected Skills", detected_value, detected_desc)
	with c2:
		render_metric_card("Recommended Domain", domain_value, domain_desc)
	with c3:
		render_metric_card("Career Guidance", "Personalized", "Recommendations are tailored to your profile.")

	st.markdown('<div class="page-card">', unsafe_allow_html=True)
	left, right = st.columns(2, gap="large")
	with left:
		render_section_title("Detected Skills", "These are the skills currently recognized from your profile.")
		if detected:
			st.write(", ".join(detected))
		else:
			st.info("No data available yet. Start by analyzing your resume.")
	with right:
		render_section_title("Recommended Domain", "This shows the domain that best aligns with your profile.")
		if domain != "Not available yet":
			st.metric("Best Domain", domain)
		else:
			st.info("Add your skills to see recommendations.")
	st.markdown('</div>', unsafe_allow_html=True)


def show_recommendation_output(recommendations: pd.DataFrame, user_skills: list[str], jobs_df: pd.DataFrame) -> None:
	if recommendations.empty:
		st.warning("No jobs found for the selected filter.")
		return
	st.markdown('<div class="page-card">', unsafe_allow_html=True)
	render_section_title("Job Recommendations", "Best matching roles based on your profile.")
	for _, row in recommendations.iterrows():
		st.markdown(
			f"""
			<div class="soft-card" style="margin-bottom: 0.7rem; display: flex; justify-content: space-between; gap: 1rem; align-items: center;">
				<div>
					<div style="font-weight: 800; font-size: 1rem;">{row['Job Role']}</div>
					<div style="color: var(--muted); font-size: 0.92rem; margin-top: 0.2rem;">{row['Domain']}</div>
				</div>
				<div style="text-align: right;">
					<div style="font-weight: 900; font-size: 1.2rem; color: #efe7ff;">{row['score']}%</div>
					<div style="color: var(--muted); font-size: 0.82rem;">{score_band(float(row['score']))}</div>
				</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
	related_roles = suggest_related_roles(user_skills, jobs_df, recommendations, limit=4)
	if related_roles:
		render_section_title("Related Roles", "Additional job names that fit your profile beyond the skill list.")
		role_columns = st.columns(min(2, len(related_roles)), gap="medium")
		for index, role in enumerate(related_roles):
			with role_columns[index % len(role_columns)]:
				st.markdown(
					f"""
					<div class="soft-card" style="margin-bottom: 0.7rem; min-height: 110px;">
						<div style="font-weight: 800; font-size: 1rem;">{role['Job Role']}</div>
						<div style="color: var(--muted); font-size: 0.9rem; margin-top: 0.25rem;">{role['Domain']}</div>
						<div style="margin-top: 0.45rem; font-size: 0.84rem; color: #efe7ff; line-height: 1.5;">{role['reason']}</div>
					</div>
					""",
					unsafe_allow_html=True,
				)
	top_job = recommendations.iloc[0]
	missing = skill_gap_analysis(user_skills, str(top_job["Skills"]))
	render_section_title("Skill Gap Analysis", "Skills you can improve to strengthen your profile.")
	if missing:
		st.warning("You lack: " + ", ".join(missing))
	else:
		st.success("You currently match most skills required for the top role.")
	render_section_title("Career Path Suggestion", "Your next steps to reach the recommended role.")
	st.write(career_path_message(str(top_job["Job Role"]), missing))
	render_section_title("Course Recommendation", "Recommended learning to close your skill gaps.")
	for course in recommend_courses(missing, top_job_role=str(top_job["Job Role"]), top_domain=str(top_job["Domain"])):
		st.markdown(
			f"""
			<div class="soft-card" style="margin-bottom: 0.65rem;">
				<div style="font-weight: 800; font-size: 1rem; margin-bottom: 0.25rem;">{course['course']}</div>
				<div style="color: var(--muted); font-size: 0.9rem; line-height: 1.6;">{course['why_it_helps']}</div>
				<div style="margin-top: 0.35rem; font-size: 0.88rem; color: #efe7ff;">Resume benefit: {course['resume_benefit']}</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
	st.markdown('</div>', unsafe_allow_html=True)


def page_resume_analyzer(jobs_df: pd.DataFrame) -> None:
	render_hero(
		"Resume Analyzer",
		"Upload your resume to discover matching roles, key skill gaps, and your ideal career direction.",
		kicker="Resume workflow",
	)
	st.markdown('<div class="page-card">', unsafe_allow_html=True)
	c1, c2 = st.columns([1.1, 0.9], gap="large")
	with c1:
		domain_filter = st.selectbox("Filter jobs by domain", ["All"] + DOMAINS, key="resume_domain")
	with c2:
		top_n = st.slider("Number of top jobs", 5, 8, 5, key="resume_top_n")
	resume_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
	st.markdown('</div>', unsafe_allow_html=True)
	if st.button("Analyze Resume", use_container_width=True):
		if resume_file is None:
			st.error("Please upload a PDF resume first.")
			return
		with st.spinner("Analyzing your resume..."):
			raw_text, extraction_message = extract_resume_text(resume_file)
			if not raw_text.strip():
				st.error(extraction_message or "Could not extract text from PDF. Try another file.")
				st.info("If this is a scanned resume, export it as a text-based PDF or use a resume with selectable text.")
				return
			clean_text = preprocess_text(raw_text)
			detected_skills = extract_skills_from_text(clean_text, COMMON_SKILLS)
			recommendations = recommend_jobs(clean_text, jobs_df, domain_filter=domain_filter, top_n=top_n)
			domain = suggest_domain(recommendations)
			time.sleep(1)
		st.markdown('<div class="page-card">', unsafe_allow_html=True)
		render_section_title("Detected Skills", "Skills identified from your resume.")
		if detected_skills:
			st.write(", ".join(detected_skills))
		else:
			st.info("No skills were identified from this resume yet.")
		st.markdown('</div>', unsafe_allow_html=True)
		show_recommendation_output(recommendations, detected_skills, jobs_df)
		st.session_state.detected_skills = detected_skills
		st.session_state.recommended_domain = domain
		append_history(st.session_state.username, "Resume", st.session_state.name, clean_text, detected_skills, domain, recommendations)


def page_manual_mode(jobs_df: pd.DataFrame) -> None:
	render_hero(
		"Manual Input Mode",
		"Enter your details to receive personalized job recommendations and career insights.",
		kicker="Manual Input",
	)
	st.markdown('<div class="page-card">', unsafe_allow_html=True)
	with st.form("manual_form"):
		grid1, grid2 = st.columns(2, gap="large")
		with grid1:
			name = st.text_input("Name", value=st.session_state.name)
			degree = st.selectbox("Degree", ["B.Tech", "BBA", "BA", "B.Com", "MBA", "M.Tech", "Other"])
			cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
			domain_filter = st.selectbox("Choose a domain (optional)", ["All"] + DOMAINS, key="manual_domain")
		with grid2:
			skills = st.text_area("Skills (e.g., Python, SQL, Machine Learning)", placeholder="python, sql, machine learning")
			interests = st.text_input("Interests (optional)", placeholder="AI, finance, design")
			experience = st.text_area("Experience (internships, projects, certifications)", placeholder="Internship, projects, certifications")
			top_n = st.slider("Number of recommendations", 5, 8, 6, key="manual_top_n")
		submitted = st.form_submit_button("Get Recommendations", use_container_width=True)
	st.markdown('</div>', unsafe_allow_html=True)
	if submitted:
		with st.spinner("Generating recommendations..."):
			profile = f"{name} {degree} cgpa {cgpa} skills {skills} interests {interests} experience {experience}"
			clean_text = preprocess_text(profile)
			skills_input = [s.strip().lower() for s in skills.split(",") if s.strip()]
			detected_skills = sorted(set(skills_input + extract_skills_from_text(clean_text, COMMON_SKILLS)))
			recommendations = recommend_jobs(clean_text, jobs_df, domain_filter=domain_filter, top_n=top_n)
			domain = suggest_domain(recommendations)
			time.sleep(1)
		st.markdown('<div class="page-card">', unsafe_allow_html=True)
		render_section_title("Detected Skills", "Skills identified from your details.")
		if detected_skills:
			st.write(", ".join(detected_skills))
		else:
			st.info("No data available yet. Add your skills to see recommendations.")
		render_section_title("Domain Suggestion", "The career domain most aligned with your profile.")
		st.success(domain)
		st.markdown('</div>', unsafe_allow_html=True)
		show_recommendation_output(recommendations, detected_skills, jobs_df)
		st.session_state.detected_skills = detected_skills
		st.session_state.recommended_domain = domain
		append_history(st.session_state.username, "Manual", name, clean_text, detected_skills, domain, recommendations)


def page_insights(jobs_df: pd.DataFrame) -> None:
	render_hero(
		"Insights",
		"Explore key trends from job data and your activity.",
		kicker="Analytics",
	)
	st.markdown('<div class="page-card">', unsafe_allow_html=True)
	render_section_title("Job Distribution by Domain")
	domain_counts = jobs_df["Domain"].value_counts()
	fig1, ax1 = plt.subplots(figsize=(7, 3.5))
	ax1.bar(domain_counts.index, domain_counts.values, color="#a855f7")
	ax1.set_xlabel("Domain")
	ax1.set_ylabel("Number of Roles")
	ax1.set_title("Job Distribution")
	fig1.tight_layout()
	st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
	st.pyplot(fig1, use_container_width=False)
	st.markdown('</div>', unsafe_allow_html=True)
	render_section_title("Skill Frequency")
	skills_series = jobs_df["Skills"].str.lower().str.split().explode()
	top_skills = skills_series.value_counts().head(12)
	fig2, ax2 = plt.subplots(figsize=(7, 3.5))
	ax2.bar(top_skills.index, top_skills.values, color="#7c3aed")
	ax2.set_xticklabels(top_skills.index, rotation=45, ha="right")
	ax2.set_title("Top Skills in Job Dataset")
	fig2.tight_layout()
	st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
	st.pyplot(fig2, use_container_width=False)
	st.markdown('</div>', unsafe_allow_html=True)
	render_section_title("Domain Popularity")
	hist = read_history(st.session_state.username)
	if hist.empty:
		st.info("No data available yet. Start by analyzing your resume.")
		st.markdown('</div>', unsafe_allow_html=True)
		return
	pop = hist["domain_suggestion"].value_counts()
	fig3, ax3 = plt.subplots(figsize=(6, 3))
	ax3.pie(pop.values, labels=pop.index, autopct="%1.1f%%", startangle=90)
	ax3.set_title("Recommended Domain Popularity")
	fig3.tight_layout()
	st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
	st.pyplot(fig3, use_container_width=False)
	st.markdown('</div>', unsafe_allow_html=True)
	st.markdown('</div>', unsafe_allow_html=True)


def page_feedback() -> None:
	render_hero(
		"Feedback",
		"Share a quick rating and comments so the experience can keep improving.",
		kicker="User voice",
	)
	st.markdown('<div class="page-card">', unsafe_allow_html=True)
	rating = st.slider("Rate this system", 1, 5, 4)
	feedback_text = st.text_area("Write your feedback")
	if st.button("Submit Feedback", use_container_width=True):
		save_feedback(st.session_state.username, rating, feedback_text)
		st.success("Thanks for your feedback")
	st.markdown('</div>', unsafe_allow_html=True)


def page_history() -> None:
	render_hero(
		"History",
		"Review resume and manual actions, job results, match scores, and timestamps in one place.",
		kicker="Activity log",
	)
	st.markdown('<div class="page-card">', unsafe_allow_html=True)
	history_df = read_history(st.session_state.username)
	if history_df.empty:
		st.info("No history records available yet.")
	else:
		mode_filter = st.selectbox("Filter by mode", ["All", "Resume", "Manual"])
		view_df = history_df.copy()
		if mode_filter != "All":
			view_df = view_df[view_df["mode"] == mode_filter]
		def _safe_parse(val: str):
			try:
				return ast.literal_eval(val)
			except Exception:
				return val
		view_df["top_jobs"] = view_df["top_jobs"].astype(str).apply(_safe_parse)
		view_df["top_scores"] = view_df["top_scores"].astype(str).apply(_safe_parse)
		view_df["detected_skills"] = view_df["detected_skills"].astype(str).apply(_safe_parse)
		st.dataframe(view_df, use_container_width=True)
	if st.button("Clear History", type="secondary"):
		full = pd.read_csv(HISTORY_FILE)
		full = full[full["username"] != st.session_state.username]
		full.to_csv(HISTORY_FILE, index=False)
		st.success("History cleared")
		st.rerun()
	st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
	st.set_page_config(page_title="Edu2Job", layout="wide")
	inject_ui_styles()
	init_files()
	if "logged_in" not in st.session_state:
		st.session_state.logged_in = False
		st.session_state.username = ""
		st.session_state.name = ""
		st.session_state.detected_skills = []
		st.session_state.recommended_domain = "Not available yet"
	if not st.session_state.logged_in:
		auth_page()
		return
	jobs_df = load_jobs()
	st.sidebar.markdown(
		"""
		<div style="padding: 0.6rem 0.4rem 1rem;">
			<div style="font-size: 1.05rem; font-weight: 800; letter-spacing: -0.02em;">Navigation</div>
			<div style="color: var(--muted); font-size: 0.9rem; margin-top: 0.3rem;">Choose a section to continue your career journey.</div>
		</div>
		""",
		unsafe_allow_html=True,
	)
	st.sidebar.markdown(
		f"""
		<div class="soft-card" style="margin-bottom: 0.9rem;">
			<div style="font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em;">Signed in as</div>
			<div style="font-size: 1rem; font-weight: 800; margin-top: 0.25rem;">{st.session_state.name}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)
	page = st.sidebar.radio("Go to", ["Dashboard", "Resume Analyzer", "Manual Input", "Insights", "Feedback", "History"])
	if st.sidebar.button("Logout"):
		st.session_state.logged_in = False
		st.rerun()
	if page == "Dashboard":
		page_dashboard()
	elif page == "Resume Analyzer":
		page_resume_analyzer(jobs_df)
	elif page == "Manual Input":
		page_manual_mode(jobs_df)
	elif page == "Insights":
		page_insights(jobs_df)
	elif page == "Feedback":
		page_feedback()
	elif page == "History":
		page_history()


if __name__ == "__main__":
	main()

