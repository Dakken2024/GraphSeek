"""
Agent implementation with ReAct reasoning framework.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import json
import re

from tools.base import ToolRegistry, ToolResult


@dataclass
class AgentState:
    """State container for agent execution."""
    query: str = ""
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    current_step: int = 0
    thoughts: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Any] = field(default_factory=list)
    final_answer: Optional[str] = None
    max_iterations: int = 10
    current_iteration: int = 0
    
    def add_thought(self, thought: str) -> None:
        """Add a thought to the reasoning chain."""
        self.thoughts.append(thought)
    
    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Record a tool call."""
        self.tool_calls.append({
            "tool": tool_name,
            "arguments": arguments,
        })
    
    def add_observation(self, observation: Any) -> None:
        """Add an observation from tool execution."""
        self.observations.append(observation)
    
    def is_finished(self) -> bool:
        """Check if agent has reached a final answer."""
        return self.final_answer is not None
    
    def should_continue(self) -> bool:
        """Check if agent should continue iterating."""
        return (
            not self.is_finished() 
            and self.current_iteration < self.max_iterations
        )


class Agent(ABC):
    """Abstract base class for agents."""
    
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
    
    @abstractmethod
    def run(self, query: str, chat_history: List[Dict[str, str]] = None) -> AgentState:
        """Run the agent on a query."""
        pass
    
    @abstractmethod
    def _reason(self, state: AgentState) -> str:
        """Generate next reasoning step."""
        pass


class ReActAgent(Agent):
    """ReAct (Reasoning + Acting) agent implementation."""
    
    def __init__(
        self, 
        tool_registry: ToolRegistry,
        llm_service=None,
        system_prompt: Optional[str] = None,
    ) -> None:
        super().__init__(tool_registry)
        self.llm_service = llm_service
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Get default system prompt for ReAct reasoning."""
        return """You are an intelligent assistant that uses a ReAct (Reasoning + Acting) approach to answer questions.

You have access to the following tools:
{tools}

Use the following format:

Thought: You should always think about what to do
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: The final answer to the original input question

Begin!

Question: {query}
{chat_history}
{agent_scratchpad}"""
    
    def _format_tools_description(self) -> str:
        """Format tool descriptions for the prompt."""
        tools_desc = []
        for tool_def in self.tool_registry.get_all_definitions():
            tools_desc.append(f"- {tool_def.name}: {tool_def.description}")
        return "\n".join(tools_desc)
    
    def _format_tool_names(self) -> str:
        """Get comma-separated list of tool names."""
        return ", ".join(self.tool_registry.list_tools())
    
    def _parse_action(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse action from LLM response."""
        # Look for Action: and Action Input: patterns
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        action_input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            try:
                # Try to parse as JSON first
                action_input = json.loads(action_input_match.group(1).strip())
            except json.JSONDecodeError:
                # Fall back to simple key=value parsing
                action_input_str = action_input_match.group(1).strip()
                action_input = {"query": action_input_str}
            
            return {"action": action, "action_input": action_input}
        
        return None
    
    def _parse_final_answer(self, response: str) -> Optional[str]:
        """Parse final answer from LLM response."""
        final_match = re.search(r"Final Answer:\s*(.+?)$", response, re.DOTALL | re.IGNORECASE)
        if final_match:
            return final_match.group(1).strip()
        return None
    
    def run(self, query: str, chat_history: List[Dict[str, str]] = None) -> AgentState:
        """Run the ReAct agent on a query."""
        state = AgentState(
            query=query,
            chat_history=chat_history or [],
        )
        
        # Format chat history
        chat_history_str = ""
        if state.chat_history:
            chat_history_str = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in state.chat_history[-5:]
            ])
        
        agent_scratchpad = ""
        
        while state.should_continue():
            state.current_iteration += 1
            
            # Build prompt
            prompt = self.system_prompt.format(
                tools=self._format_tools_description(),
                tool_names=self._format_tool_names(),
                query=query,
                chat_history=chat_history_str,
                agent_scratchpad=agent_scratchpad,
            )
            
            # Get LLM response (non-streaming for agent reasoning)
            if self.llm_service:
                tokens = []
                for token in self.llm_service.generate_response(
                    prompt=prompt,
                    temperature=0.0,  # Lower temperature for more deterministic reasoning
                    max_context=4096,
                ):
                    tokens.append(token)
                response = "".join(tokens)
            else:
                response = "Thought: I need to use tools but no LLM service is available."
            
            # Parse response
            state.add_thought(response)
            
            # Check for final answer
            final_answer = self._parse_final_answer(response)
            if final_answer:
                state.final_answer = final_answer
                break
            
            # Check for action
            action_data = self._parse_action(response)
            if action_data:
                action_name = action_data["action"]
                action_input = action_data["action_input"]
                
                state.add_tool_call(action_name, action_input)
                
                # Execute tool
                result = self.tool_registry.execute(action_name, **action_input)
                state.add_observation(result.data if result.success else result.error)
                
                # Update scratchpad
                agent_scratchpad += f"""
Thought: {response.split('Thought:')[-1].split('Action:')[0] if 'Thought:' in response else ''}
Action: {action_name}
Action Input: {json.dumps(action_input)}
Observation: {result.data if result.success else f'Error: {result.error}'}
"""
            else:
                # No clear action, add response to scratchpad
                agent_scratchpad += f"\n{response}\n"
        
        # If no final answer was reached, provide best effort response
        if not state.final_answer:
            if state.observations:
                state.final_answer = (
                    f"Based on my analysis:\n"
                    f"{' '.join(str(obs) for obs in state.observations[:3])}"
                )
            else:
                state.final_answer = "Unable to determine a final answer."
        
        return state
    
    def run_streaming(self, query: str, chat_history: List[Dict[str, str]] = None):
        """Run the agent with streaming responses (generator)."""
        state = AgentState(
            query=query,
            chat_history=chat_history or [],
        )
        
        chat_history_str = ""
        if state.chat_history:
            chat_history_str = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in state.chat_history[-5:]
            ])
        
        agent_scratchpad = ""
        
        yield {"type": "start", "query": query}
        
        while state.should_continue():
            state.current_iteration += 1
            
            prompt = self.system_prompt.format(
                tools=self._format_tools_description(),
                tool_names=self._format_tool_names(),
                query=query,
                chat_history=chat_history_str,
                agent_scratchpad=agent_scratchpad,
            )
            
            yield {"type": "thought_start", "iteration": state.current_iteration}
            
            # Stream thought generation
            thought_content = ""
            if self.llm_service:
                for token in self.llm_service.generate_response(
                    prompt=prompt,
                    temperature=0.0,
                    max_context=4096,
                ):
                    thought_content += token
                    yield {"type": "thought_token", "token": token}
            else:
                thought_content = "No LLM service available."
            
            state.add_thought(thought_content)
            yield {"type": "thought_complete", "content": thought_content}
            
            # Check for final answer
            final_answer = self._parse_final_answer(thought_content)
            if final_answer:
                state.final_answer = final_answer
                yield {"type": "final_answer", "answer": final_answer}
                break
            
            # Check for action
            action_data = self._parse_action(thought_content)
            if action_data:
                action_name = action_data["action"]
                action_input = action_data["action_input"]
                
                state.add_tool_call(action_name, action_input)
                yield {"type": "tool_call", "tool": action_name, "input": action_input}
                
                # Execute tool
                result = self.tool_registry.execute(action_name, **action_input)
                observation = result.data if result.success else result.error
                state.add_observation(observation)
                
                yield {"type": "tool_result", "success": result.success, "data": observation}
                
                agent_scratchpad += f"""
{thought_content}
Observation: {observation}
"""
            else:
                agent_scratchpad += f"\n{thought_content}\n"
        
        if not state.final_answer:
            if state.observations:
                state.final_answer = f"Analysis: {' '.join(str(obs) for obs in state.observations[:3])}"
            else:
                state.final_answer = "Unable to determine answer."
            
            yield {"type": "final_answer", "answer": state.final_answer}
        
        yield {"type": "complete", "state": state}
