# Interview Preparation Agentic Tool

An intelligent CLI tool that helps you prepare for interviews by generating personalized questions and providing feedback on your answers. Built with **LlamaIndex Workflows** for true agentic processing, LlamaParse for advanced document processing, and OpenAI for intelligent question generation and feedback.

## Features

- 🤖 **Agentic Workflow Architecture**: Built using LlamaIndex Workflows with proper event-driven processing
- 📄 **CV Parsing**: Automatically extracts text from PDF CVs using LlamaParse
- 🔒 **Privacy Protection**: Anonymizes company names and personal information
- 🎯 **Smart Question Generation**: Creates tailored interview questions based on your background, job requirements, and interviewer profile
- 💡 **AI Feedback**: Provides constructive feedback on your answers using advanced LLM analysis
- 🔧 **LlamaIndex Integration**: Leverages semantic search and retrieval for contextual understanding
- 👤 **Human-in-the-Loop**: Proper workflow integration for interactive user input

## Prerequisites

Before using this tool, you'll need:

1. **OpenAI API Key** - For question generation and feedback
2. **LlamaParse API Key** - For PDF parsing (get it from [LlamaIndex](https://cloud.llamaindex.ai/))
3. **Python 3.8+**

## Installation

1. **Clone or download this repository**
   ```bash
   cd interview_prep
   ```

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

## Privacy and Anonymization

The tool automatically anonymizes:
- Personal names → `[NAME]`
- Company names → `[COMPANY]`
- Email addresses → `[EMAIL]`
- Phone numbers → `[PHONE]`

This ensures your sensitive information is protected during processing.

## Architecture

The tool uses a sophisticated **LlamaIndex Workflows** architecture:

### Workflow Steps:
1. **CV Parsing Step**: Parses PDF and anonymizes content using LlamaParse
2. **Job Description Loading Step**: Loads and processes job requirements
3. **Interviewer Info Loading Step**: Loads interviewer background information
4. **Question Generation Step**: Creates personalized questions using semantic analysis
5. **Human-in-the-Loop Step**: Captures user answer through InputRequiredEvent
6. **Feedback Generation Step**: Analyzes answer and provides constructive feedback

### Core Technologies:
- **LlamaIndex Workflows**: Event-driven agentic processing with StartEvent, StopEvent, and custom events
- **LlamaParse**: Advanced PDF parsing with structure preservation
- **LlamaIndex**: Document indexing and semantic retrieval
- **OpenAI GPT-4**: Question generation and feedback analysis
- **Click**: User-friendly CLI interface
- **Human-in-the-Loop Events**: InputRequiredEvent and HumanResponseEvent for interactive processing

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure both `OPENAI_API_KEY` and `LLAMA_PARSE_API_KEY` are set correctly
   - Check that your OpenAI account has sufficient credits

2. **PDF Parsing Issues**
   - Ensure the CV file exists and is a valid PDF
   - LlamaParse works best with text-based PDFs (not scanned images)

3. **File Not Found Errors**
   - Check file paths are correct and files exist
   - Use absolute paths if relative paths don't work

### Getting API Keys

1. **OpenAI API Key**:
   - Go to [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create an account and generate an API key

2. **LlamaParse API Key**:
   - Visit [LlamaIndex Cloud](https://cloud.llamaindex.ai/)
   - Sign up and get your API key from the dashboard

## Future Enhancements

Potential improvements leveraging the workflow architecture:
- **Multi-Round Interviews**: Extend workflow with multiple question/feedback cycles
- **Adaptive Questioning**: Dynamic question generation based on previous answers
- **Web interface for easier file uploads**: Convert CLI to web-based interface while maintaining workflow backend
- **Interview session recording and analysis**: Add workflow steps for session persistence
- **Company-specific question templates**: Create specialized workflow branches for different company types
- **Integration with job boards**: Add workflow steps for automatic job description fetching
- **Parallel Processing**: Use workflow's event system for concurrent question generation
- **Custom Agent Integration**: Add specialized agents for different interview types (technical, behavioral, etc.)

## Support

If you encounter issues:
1. Check that all dependencies are installed correctly
2. Verify API keys are valid and have sufficient credits
3. Ensure input files are properly formatted
4. Review error messages for specific guidance

---

Happy interviewing! 🎯 