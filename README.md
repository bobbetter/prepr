# Interview Preparation Agentic Tool

An intelligent CLI tool that helps you prepare for interviews by generating personalized questions and providing feedback on your answers. Built with **LlamaIndex Workflows** for true agentic processing, LlamaParse for advanced document processing, and OpenAI for intelligent question generation and feedback.


## Prerequisites

Before using this tool, you'll need:

1. **OpenAI API Key** - For question generation and feedback
2. **LlamaParse API Key** - For PDF parsing (get it from [LlamaIndex](https://cloud.llamaindex.ai/))
3. **mem0 API Key** - To give the agent ability to remember previously asked questions / uploaded document

## Installation

1. **Clone or download this repository**


2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup environment variables**
   - Copy `.env.example` to `.env`
   - Add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   LLAMA_PARSE_API_KEY=your_llama_parse_api_key_here
   MEM0_API_KEY=your_mem0_api_key_here
   ```

## Usage

### Running the Tool

```bash
python main.py
```

### Required Files

Before running, prepare these files:

1. **CV PDF file** - Your resume in PDF format
2. **Job description text file** - The job posting/requirements (save as `.txt`)
3. **Interviewer information text file** - Details about the interviewer's background, role, style (save as `.txt`)

### Example Workflow

1. **Start the tool**:
   ```bash
   python main.py
   ```

2. **Provide file paths** when prompted:
   ```
   📄 Enter the path to your CV PDF file: ./my_resume.pdf
   📋 Enter the path to the job description text file: ./job_description.txt
   👤 Enter the path to the interviewer information text file: ./interviewer_info.txt
   ```

3. **Receive a personalized question** based on your profile

4. **Answer the question** when prompted

5. **Get detailed feedback** on your response

### Sample File Contents

**job_description.txt**:
```
Senior Software Engineer - Backend Development
We are looking for an experienced backend developer with expertise in:
- Python and Django/Flask
- Microservices architecture
- AWS cloud services
- Database design and optimization
- API development and documentation
```

**interviewer_info.txt**:
```
Interviewer: Technical Lead with 8 years experience
Background: Former startup CTO, now at a mid-size tech company
Interview style: Focuses on problem-solving and system design
Prefers practical examples over theoretical knowledge
Values clean code and scalability discussions
```
