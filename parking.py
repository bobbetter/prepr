    def provide_feedback(self, answer: str) -> str:
        """Provide feedback on the user's answer to the current question."""
        
        if "current_question" not in self.context:
            return "❌ No current question to provide feedback on. Please generate a question first."
        
        try:
            print("\n🔄 Analyzing your answer...")
            
            # Create comprehensive prompt with all context
            full_prompt = f"""
            You are an expert interviewer providing constructive feedback on interview answers.
            
            INTERVIEW QUESTION: {self.context['current_question']}
            
            CANDIDATE'S ANSWER: {answer}
            
            CANDIDATE'S CV (Anonymized):
            {self.context.get(ContextType.CV.value, 'Not available')}
            
            JOB DESCRIPTION:
            {self.context.get(ContextType.JOB_DESCRIPTION.value, 'Not available')}
            
            INTERVIEWER INFORMATION:
            {self.context.get(ContextType.INTERVIEWER_INFO.value, 'Not available')}
            
            TASK:
            Please provide constructive feedback on this interview answer. Consider:
            1. How well the answer addresses the question
            2. Technical accuracy (if applicable)
            3. Communication clarity
            4. Areas for improvement
            5. What the answer demonstrates about the candidate's fit for the role
            6. Specific suggestions for improvement
            
            Provide specific, actionable feedback that would help the candidate improve.
            """
            
            # Generate feedback using direct LLM call
            response = self.llm.complete(full_prompt)
            feedback = str(response).strip()
            
            # Store the answer and feedback
            self.context["last_answer"] = answer
            self.context["last_feedback"] = feedback
            
            return f"📝 FEEDBACK:\n\n{feedback}\n\n🎉 Feedback complete! You can ask for another question or more specific feedback."
            
        except Exception as e:
            return f"❌ Error providing feedback: {str(e)}"