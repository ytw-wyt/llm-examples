# Lesson generator for tutors

## Overview of the App

This app is used to generate lessons for tutors by using LLM.

Input need to be entered before creating a lesson: 
- topic
- learning objective()
- OpenAI key
- PDFs as references for generation

then clicking the generate button will lead to the lesson.

## Run it locally

```sh
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run step1.py
git checkout dev
```
* The last step is because this code is still in the developing mode

References: Streamlit + LLM Examples App
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)
