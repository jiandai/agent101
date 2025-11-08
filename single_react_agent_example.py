
import os
import json
import re
import time
from enum import Enum
from typing import List, Dict, Any, Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()


# ============= SYSTEM MESSAGE MODULE =============

class SystemMessage:
    """Manages system prompts for the agent."""
    
    def __init__(self, prompt: Optional[str] = None):
        """
        Initialize SystemMessage with a prompt.
        
        Args:
            prompt: Optional system prompt. If None, uses default customer support prompt.
        """
        if prompt is None:
            prompt = """You are a helpful customer support agent. 
        
Your job is to help customers with their orders. You have access to tools 
that let you check order status, track shipments, and process refunds.

IMPORTANT: You MUST use tools to gather information before responding. 
Do NOT provide text responses until you have retrieved all necessary information using tools.
Do NOT give preliminary acknowledgments or say things like "I'll help you" or "Let me check".
Instead, immediately use the appropriate tools to gather the required information.

Think step by step:
1. Understand what the customer needs
2. Determine what information you need to help them
3. Use the appropriate tools to gather that information FIRST
4. Only after you have all the information, provide a clear, helpful text response

Always be friendly and professional."""
        
        self.prompt = prompt
        self.version = "1.0"
    
    def get_prompt(self) -> str:
        """Get the current system prompt."""
        return self.prompt
    
    def update_prompt(self, new_prompt: str, version: Optional[str] = None):
        """
        Update the system prompt.
        
        Args:
            new_prompt: New system prompt text
            version: Optional version identifier
        """
        self.prompt = new_prompt
        if version:
            self.version = version
    
    def get_version(self) -> str:
        """Get the current prompt version."""
        return self.version


# ============= MEMORY BANK MODULE =============

class MemoryBank:
    """Manages conversation history and memory for the agent."""
    
    def __init__(self):
        """Initialize an empty memory bank."""
        self.history: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: Any):
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content (string or list for Anthropic format)
        """
        self.history.append({
            "role": role,
            "content": content
        })
    
    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: Any):
        """Add an assistant message to history (can be text or tool_use blocks)."""
        self.add_message("assistant", content)
    
    def add_tool_result(self, tool_use_id: str, result: Dict[str, Any]):
        """
        Add a tool result to history.
        
        Args:
            tool_use_id: The ID of the tool use this result corresponds to
            result: The tool execution result
        """
        self.add_message("user", [{
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": json.dumps(result)
        }])
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return self.history.copy()
    
    def clear(self):
        """Clear all conversation history."""
        self.history = []
    
    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the last message in history."""
        return self.history[-1] if self.history else None


# ============= TOOLKIT MODULE =============

class Toolkit:
    """Manages tool definitions and execution for the agent."""
    
    def __init__(self):
        """Initialize toolkit with default tools."""
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools for customer support."""
        default_tools = [
            {
                "name": "get_order",
                "description": "Retrieves order information from the database",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order ID (e.g., ORD-12345)"
                        }
                    },
                    "required": ["order_id"]
                }
            },
            {
                "name": "track_shipment",
                "description": "Gets real-time tracking information for a package",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "tracking_number": {
                            "type": "string",
                            "description": "The tracking number"
                        }
                    },
                    "required": ["tracking_number"]
                }
            },
            {
                "name": "process_refund",
                "description": "Initiates a refund for an order",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order ID to refund"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for refund"
                        }
                    },
                    "required": ["order_id"]
                }
            }
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool_definition: Dict[str, Any]):
        """
        Register a new tool.
        
        Args:
            tool_definition: Tool definition dictionary with name, description, and input_schema
        """
        tool_name = tool_definition.get("name")
        if not tool_name:
            raise ValueError("Tool definition must include a 'name' field")
        
        self.tools[tool_name] = tool_definition
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get all registered tool definitions as a list."""
        return list(self.tools.values())
    
    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            
        Returns:
            Tool execution result dictionary
        """
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        # Mock tool implementations
        if tool_name == "get_order":
            return {
                "order_id": tool_input["order_id"],
                "status": "shipped",
                "tracking_number": "TRK-98765",
                "estimated_delivery": "2025-11-04"
            }
        elif tool_name == "track_shipment":
            return {
                "status": "Out for delivery",
                "location": "Local Distribution Center",
                "expected_delivery": "Today by 8:00 PM"
            }
        elif tool_name == "process_refund":
            return {
                "refund_id": "REF-" + tool_input["order_id"],
                "status": "initiated",
                "processing_time": "5-7 business days"
            }
        else:
            return {"error": "Tool not implemented"}


# ============= LLM CLIENT MODULE =============

class LLMClient:
    """Manages LLM API interactions with error handling and response parsing."""
   
    def __init__(self, provider="anthropic", model="claude-haiku-4-5-20251001"):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider name (default: "anthropic")
            model: Model name to use
        """
        self.provider = provider
        self.model = model
        self.api_calls_made = []
        self._client = None
    
    def _get_client(self):
        """Get or create the API client."""
        if self._client is None:
            if self.provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_CLAUDE_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_CLAUDE_API_KEY environment variable not set")
                self._client = anthropic.Anthropic(api_key=api_key)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client
    
    def call_llm(self, messages: List[Dict], tools: List[Dict] = None, system: str = None):
        """
        Call the LLM API with messages, tools, and system prompt.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            system: Optional system prompt string
            
        Returns:
            LLM API response object
            
        Raises:
            Exception: If API call fails
        """
        try:
            client = self._get_client()
            api_params = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": messages,
            }
            if tools:
                api_params["tools"] = tools
            if system:
                api_params["system"] = system
            
            response = client.messages.create(**api_params)
            
            # Track that we made a call
            self.api_calls_made.append({
                "model": self.model,
                "messages": messages,
                "tools": [t["name"] for t in tools] if tools else []
            })
            
            return response
        except Exception as e:
            print(f"⚠️  LLM API call failed: {str(e)}")
            raise
    
    def parse_response(self, response) -> Dict[str, Any]:
        """
        Parse LLM response into structured format.
        
        Args:
            response: Raw LLM API response
            
        Returns:
            Dictionary with parsed response information
        """
        if not response or not hasattr(response, 'content') or not response.content:
            return {"type": "empty", "content": None}
        
        first_block = response.content[0]
        response_type = first_block.type
        
        parsed = {
            "type": response_type,
            "raw": response,
            "blocks": response.content
        }
        
        if response_type == "tool_use":
            parsed["tool_name"] = first_block.name
            parsed["tool_input"] = first_block.input
            parsed["tool_use_id"] = first_block.id
        elif response_type == "text":
            text_blocks = [block for block in response.content if block.type == "text"]
            parsed["text"] = " ".join(block.text for block in text_blocks) if text_blocks else ""
            tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
            parsed["tool_use_blocks"] = tool_use_blocks
        
        return parsed
    
    def get_api_call_count(self) -> int:
        """Get the total number of API calls made."""
        return len(self.api_calls_made)


# ============= AGENT STATE ENUM =============

class AgentState(Enum):
    """Agent execution states."""
    INIT = "initializing"
    UNDERSTANDING = "understanding_goal"
    PLANNING = "creating_plan"
    EXECUTING = "executing_plan"
    SYNTHESIZING = "creating_response"
    VERIFYING = "verifying_completion"
    COMPLETE = "complete"
    FAILED = "failed"


# ============= AGENT CLASS =============

class ReActAgent:
    """Single agent implementation using modular framework components."""
    
    def __init__(self, 
                 llm_client: Optional[LLMClient] = None,
                 toolkit: Optional[Toolkit] = None,
                 memory_bank: Optional[MemoryBank] = None,
                 system_message: Optional[SystemMessage] = None):
        """
        Initialize the agent with modular components.
        
        Args:
            llm_client: Optional LLMClient instance (creates default if None)
            toolkit: Optional Toolkit instance (creates default if None)
            memory_bank: Optional MemoryBank instance (creates default if None)
            system_message: Optional SystemMessage instance (creates default if None)
        """
        # Initialize all framework modules
        self.llm = llm_client if llm_client is not None else LLMClient()
        self.toolkit = toolkit if toolkit is not None else Toolkit()
        self.memory = memory_bank if memory_bank is not None else MemoryBank()
        self.system_message = system_message if system_message is not None else SystemMessage()
        
        # State machine
        self.state = AgentState.INIT
        self.execution_context = {
            "goal": None,
            "plan": None,
            "results": {},
            "attempts": 0
        }
    
    def transition_to(self, new_state: AgentState):
        """
        Transition to new state with logging.
        
        Args:
            new_state: The new state to transition to
        """
        print(f"🔄 State transition: {self.state.value} → {new_state.value}")
        self.state = new_state
    
    def _extract_goal_with_llm(self, customer_message: str) -> Dict[str, Any]:
        """
        Use LLM to extract structured goal from customer message.
        
        Args:
            customer_message: The customer's message
            
        Returns:
            Dictionary with goal information (backward compatible structure)
        """
        extraction_prompt = f"""Analyze this customer message and extract structured information:

Customer message: "{customer_message}"

Extract the following information and return as JSON:
{{
    "needs_order_info": boolean - true if customer needs order status/information,
    "needs_tracking": boolean - true if customer needs tracking information,
    "needs_refund": boolean - true if customer wants refund/return/cancel,
    "order_id": string or null - any order ID mentioned (e.g., ORD-12345),
    "tracking_number": string or null - any tracking number mentioned (e.g., TRK-98765),
    "key_phrases": array of strings - important phrases from the message,
    "core_request": string - brief summary of what the customer wants
}}

Return ONLY valid JSON, no other text."""

        try:
            response = self.llm.call_llm(
                messages=[{"role": "user", "content": extraction_prompt}],
                system="You are a goal extraction system. Analyze customer messages and extract structured information. Return ONLY valid JSON, no explanations or markdown."
            )
            
            # Extract text from response
            parsed = self.llm.parse_response(response)
            if parsed["type"] == "text":
                text_content = parsed["text"]
                # Try to extract JSON from response (handle markdown code blocks)
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text_content, re.DOTALL)
                if json_match:
                    goal = json.loads(json_match.group(0))
                else:
                    # Try parsing entire text as JSON
                    goal = json.loads(text_content)
                
                # Ensure backward compatibility with expected structure
                if "key_phrases" not in goal:
                    goal["key_phrases"] = []
                if "core_request" not in goal:
                    goal["core_request"] = customer_message[:100]
                
                return goal
            else:
                raise ValueError("Unexpected response type from goal extraction")
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"⚠️  Goal extraction failed: {str(e)}, falling back to regex")
            # Fallback to regex-based extraction for backward compatibility
            return self._extract_goal_fallback(customer_message)
    
    def _extract_goal_fallback(self, customer_message: str) -> Dict[str, Any]:
        """
        Fallback regex-based goal extraction (for error recovery).
        
        Args:
            customer_message: The customer's message
            
        Returns:
            Dictionary with goal information
        """
        message_lower = customer_message.lower()
        goal = {
            "needs_order_info": False,
            "needs_tracking": False,
            "needs_refund": False,
            "order_id": None,
            "tracking_number": None,
            "key_phrases": [],
            "core_request": customer_message[:100]
        }
        
        # Extract order ID if present
        order_match = re.search(r'\bORD-?\w+\b', customer_message, re.IGNORECASE)
        if order_match:
            goal["order_id"] = order_match.group(0).upper()
        
        # Extract tracking number if present
        tracking_match = re.search(r'\bTRK-?\w+\b', customer_message, re.IGNORECASE)
        if tracking_match:
            goal["tracking_number"] = tracking_match.group(0).upper()
        
        # Determine what information is needed
        if any(phrase in message_lower for phrase in ["where", "location", "status", "when", "arrive", "deliver"]):
            goal["needs_order_info"] = True
            goal["key_phrases"].extend(["where", "location", "status", "when", "arrive", "deliver"])
        
        if any(phrase in message_lower for phrase in ["track", "tracking", "shipment", "shipping"]):
            goal["needs_tracking"] = True
            goal["key_phrases"].extend(["track", "tracking", "shipment", "shipping"])
        
        if any(phrase in message_lower for phrase in ["refund", "return", "cancel"]):
            goal["needs_refund"] = True
            goal["key_phrases"].extend(["refund", "return", "cancel"])
        
        return goal
    
    def _verify_goal_fulfillment_with_llm(self, response: str, goal: Dict[str, Any], tools_used: List[str] = None) -> bool:
        """
        Use LLM to verify if response fully addresses the goal.
        
        Args:
            response: The agent's text response
            goal: The goal dictionary extracted from customer message
            tools_used: Optional list of tools that were used
            
        Returns:
            True if goal is fulfilled, False otherwise
        """
        tools_used_str = ", ".join(tools_used) if tools_used else "none"
        
        verification_prompt = f"""Analyze if this response COMPLETELY fulfills the customer's goal.

Customer Goal:
{json.dumps(goal, indent=2)}

Agent Response:
"{response}"

Tools Used: {tools_used_str}

Does this response COMPLETELY fulfill the goal? Consider:
- Does it provide all requested information?
- Is it a complete answer (not just an acknowledgment like "I'll help you" or "Let me check")?
- Does it address the core request?

Return your analysis as JSON:
{{
    "fulfilled": true or false,
    "reasoning": "brief explanation",
    "missing_elements": ["list", "of", "missing", "info"] or []
}}

Return ONLY valid JSON, no other text."""

        try:
            result = self.llm.call_llm(
                messages=[{"role": "user", "content": verification_prompt}],
                system="You verify if responses fulfill customer goals. Be strict and only mark as fulfilled=true if the response is complete and addresses all aspects of the goal. Return ONLY valid JSON."
            )
            
            parsed = self.llm.parse_response(result)
            if parsed["type"] == "text":
                text_content = parsed["text"]
                # Try to extract JSON from response
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text_content, re.DOTALL)
                if json_match:
                    goal_check = json.loads(json_match.group(0))
                else:
                    # Try parsing entire text as JSON
                    goal_check = json.loads(text_content)
                
                is_fulfilled = goal_check.get("fulfilled", False)
                reasoning = goal_check.get("reasoning", "")
                missing_elements = goal_check.get("missing_elements", [])
                
                print(f"   Verification reasoning: {reasoning}")
                if missing_elements:
                    print(f"   Missing elements: {missing_elements}")
                
                return is_fulfilled
            else:
                # If we can't parse, be conservative and return False
                print(f"   ⚠️  Unexpected response type from verification")
                return False
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"⚠️  Goal verification JSON parsing failed: {str(e)}, using fallback")
            # Fallback to heuristic-based verification
            return self._is_goal_fulfilled_fallback(response, goal, tools_used or [])
    
    def _is_goal_fulfilled_fallback(self, text_response: str, goal: Dict[str, Any], tools_used: List[str]) -> bool:
        """
        Fallback heuristic-based goal fulfillment check (for error recovery).
        
        Args:
            text_response: The agent's text response
            goal: The goal dictionary
            tools_used: List of tools that were used
            
        Returns:
            True if goal appears fulfilled, False otherwise
        """
        text_lower = text_response.lower()
        
        # Check for acknowledgment phrases that indicate the response is NOT final
        acknowledgment_phrases = [
            "let me", "i'll", "i will", "now let me", "let me get", 
            "let me check", "i'm going to", "i need to", "i should",
            "i'll help you", "i'll look", "i'll find", "i'll retrieve",
            "let me retrieve", "let me find", "let me look"
        ]
        
        # If response starts with acknowledgment phrase, it's not fulfilled
        for phrase in acknowledgment_phrases:
            if text_lower.startswith(phrase) or text_lower[:50].find(phrase) != -1:
                return False
        
        # Check if response contains actual information
        if goal.get("needs_order_info"):
            info_indicators = ["order", "status", "shipped", "delivered", "delivery", 
                             "location", "arrive", "arriving", "out for delivery",
                             "estimated", "expected"]
            has_info = any(indicator in text_lower for indicator in info_indicators)
            if not has_info:
                return False
        
        if goal.get("needs_tracking"):
            tracking_indicators = ["tracking", "track", "shipment", "shipping", 
                                  "location", "status", "delivery"]
            has_tracking_info = any(indicator in text_lower for indicator in tracking_indicators)
            if not has_tracking_info:
                return False
        
        # If tools were used, the response should reference the results
        if goal.get("needs_order_info") or goal.get("needs_tracking"):
            if not tools_used:
                return False
        
        return True
    
    def _synthesize_response(self, goal: Dict[str, Any], results: Dict[str, Dict[str, Any]]) -> str:
        """
        Use LLM to synthesize final response from gathered tool results.
        
        Args:
            goal: The goal dictionary
            results: Dictionary mapping tool names to their results
            
        Returns:
            Synthesized response string
        """
        synthesis_prompt = f"""Synthesize a complete, helpful response for the customer based on the gathered information.

Customer Goal:
{json.dumps(goal, indent=2)}

Gathered Information:
{json.dumps(results, indent=2)}

Create a clear, friendly response that:
- Directly addresses the customer's question
- Includes all relevant information from the tools
- Is professional and helpful
- Does NOT include acknowledgments like "I'll help you" or "Let me check"
- Provides a complete answer immediately

Response:"""

        try:
            response = self.llm.call_llm(
                messages=[{"role": "user", "content": synthesis_prompt}],
                system="You synthesize customer support responses from gathered information. Be direct, complete, and helpful. Do not include preliminary acknowledgments."
            )
            
            parsed = self.llm.parse_response(response)
            if parsed["type"] == "text":
                return parsed.get("text", "")
            else:
                return "I apologize, but I encountered an error while creating the response."
                
        except Exception as e:
            print(f"⚠️  Response synthesis failed: {str(e)}")
            # Fallback: create simple response from results
            return self._create_simple_response(goal, results)
    
    def _create_simple_response(self, goal: Dict[str, Any], results: Dict[str, Dict[str, Any]]) -> str:
        """
        Create a simple response from results (fallback).
        
        Args:
            goal: The goal dictionary
            results: Dictionary mapping tool names to their results
            
        Returns:
            Simple response string
        """
        response_parts = []
        
        if "get_order" in results:
            order_info = results["get_order"]
            response_parts.append(f"Your order {order_info.get('order_id', '')} is {order_info.get('status', 'unknown')}.")
            if order_info.get("estimated_delivery"):
                response_parts.append(f"Estimated delivery: {order_info['estimated_delivery']}.")
        
        if "track_shipment" in results:
            tracking_info = results["track_shipment"]
            response_parts.append(f"Your shipment status: {tracking_info.get('status', 'unknown')}.")
            if tracking_info.get("expected_delivery"):
                response_parts.append(f"Expected delivery: {tracking_info['expected_delivery']}.")
        
        if "process_refund" in results:
            refund_info = results["process_refund"]
            response_parts.append(f"Refund {refund_info.get('refund_id', '')} has been {refund_info.get('status', 'initiated')}.")
            if refund_info.get("processing_time"):
                response_parts.append(f"Processing time: {refund_info['processing_time']}.")
        
        if response_parts:
            return " ".join(response_parts)
        else:
            return "I've gathered the information, but I'm unable to provide a complete response at this time."
    
    def _execute_tool_with_retry(self, tool_name: str, tool_input: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Execute tool with automatic retry on failures.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tool execution result dictionary
        """
        for attempt in range(max_retries):
            try:
                result = self.toolkit.execute_tool(tool_name, tool_input)
                
                # Check if result indicates error
                if "error" in result:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"⚠️  Tool error: {result.get('error')}, retrying in {wait_time}s ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Try alternative approach on final failure
                        print(f"⚠️  Tool failed after {max_retries} attempts, trying alternative approach...")
                        return self._try_alternative_approach(tool_name, tool_input, result)
                
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️  Exception: {str(e)}, retrying in {wait_time}s ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Try alternative approach on final failure
                    print(f"⚠️  Tool exception after {max_retries} attempts, trying alternative approach...")
                    return self._try_alternative_approach(tool_name, tool_input, {"error": str(e), "recoverable": True})
        
        # Should not reach here, but return error if we do
        return {"error": "Tool execution failed after all retries", "recoverable": False}
    
    def _try_alternative_approach(self, failed_tool: str, original_input: Dict[str, Any], error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to find alternative way to achieve goal when tool fails.
        
        Args:
            failed_tool: Name of the tool that failed
            original_input: Original input parameters
            error: Error information from failed attempt
            
        Returns:
            Result from alternative approach or original error
        """
        available_tools = [t["name"] for t in self.toolkit.get_tool_definitions()]
        
        alternative_prompt = f"""A tool execution failed. Find an alternative approach to achieve the same goal.

Failed Tool: '{failed_tool}'
Original Input: {json.dumps(original_input, indent=2)}
Error: {json.dumps(error, indent=2)}

Available Tools: {available_tools}

Suggest an alternative approach:
1. Can another tool achieve the same goal?
2. Can we modify the input parameters?
3. Is there a workaround?

If you find an alternative, use the appropriate tool. Otherwise, explain why no alternative exists."""

        try:
            response = self.llm.call_llm(
                messages=[{"role": "user", "content": alternative_prompt}],
                tools=self.toolkit.get_tool_definitions(),
                system="You find alternative approaches when primary methods fail. Be creative but practical."
            )
            
            # Parse response and execute alternative if tool_use found
            parsed = self.llm.parse_response(response)
            if parsed["type"] == "tool_use":
                alt_tool_name = parsed["tool_name"]
                alt_tool_input = parsed["tool_input"]
                print(f"   🔄 Trying alternative tool: {alt_tool_name}")
                return self.toolkit.execute_tool(alt_tool_name, alt_tool_input)
            else:
                # No alternative found, return original error
                print(f"   ⚠️  No alternative approach found")
                return error
                
        except Exception as e:
            print(f"⚠️  Alternative approach failed: {str(e)}")
            return error
    
    def _create_execution_plan(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use LLM to create step-by-step execution plan.
        
        Args:
            goal: The goal dictionary extracted from customer message
            
        Returns:
            List of plan steps (each step is a dict with 'step', 'tool', 'input', etc.)
        """
        available_tools = [t["name"] for t in self.toolkit.get_tool_definitions()]
        
        planning_prompt = f"""Create a step-by-step execution plan to fulfill this customer goal.

Customer Goal:
{json.dumps(goal, indent=2)}

Available Tools: {available_tools}

Create a numbered plan with specific steps. Each step should:
- Specify which tool to use (if any)
- Specify the input parameters needed
- Explain what information it will gather

Return as JSON array of steps in this format:
[
    {{
        "step": 1,
        "action": "use tool or synthesize",
        "tool": "tool_name or null",
        "input": {{"param": "value"}} or null,
        "purpose": "what this step accomplishes"
    }},
    ...
]

Return ONLY valid JSON array, no other text."""

        try:
            response = self.llm.call_llm(
                messages=[{"role": "user", "content": planning_prompt}],
                system="You create execution plans. Be specific, sequential, and practical. Return ONLY valid JSON array."
            )
            
            parsed = self.llm.parse_response(response)
            if parsed["type"] == "text":
                text_content = parsed["text"]
                # Extract JSON array from response
                json_match = re.search(r'\[[^\]]*(?:\{[^\}]*\}[^\]]*)*\]', text_content, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group(0))
                else:
                    plan = json.loads(text_content)
                
                print(f"📋 Created execution plan with {len(plan)} steps")
                for step in plan:
                    print(f"   Step {step.get('step', '?')}: {step.get('purpose', 'N/A')}")
                
                return plan
            else:
                raise ValueError("Unexpected response type from planning")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  Planning failed: {str(e)}, creating simple plan")
            # Fallback: create simple plan based on goal
            return self._create_simple_plan(goal)
    
    def _create_simple_plan(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create a simple plan based on goal structure (fallback).
        
        Args:
            goal: The goal dictionary
            
        Returns:
            Simple plan as list of steps
        """
        plan = []
        step_num = 1
        
        if goal.get("needs_order_info") and goal.get("order_id"):
            plan.append({
                "step": step_num,
                "action": "use tool",
                "tool": "get_order",
                "input": {"order_id": goal["order_id"]},
                "purpose": f"Get order information for {goal['order_id']}"
            })
            step_num += 1
        
        if goal.get("needs_tracking") and goal.get("tracking_number"):
            plan.append({
                "step": step_num,
                "action": "use tool",
                "tool": "track_shipment",
                "input": {"tracking_number": goal["tracking_number"]},
                "purpose": f"Track shipment {goal['tracking_number']}"
            })
            step_num += 1
        
        if goal.get("needs_refund") and goal.get("order_id"):
            plan.append({
                "step": step_num,
                "action": "use tool",
                "tool": "process_refund",
                "input": {"order_id": goal["order_id"]},
                "purpose": f"Process refund for {goal['order_id']}"
            })
            step_num += 1
        
        plan.append({
            "step": step_num,
            "action": "synthesize",
            "tool": None,
            "input": None,
            "purpose": "Synthesize final response from gathered information"
        })
        
        return plan
    
    def _validate_plan(self, plan: List[Dict[str, Any]], goal: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate plan before execution.
        
        Args:
            plan: The execution plan to validate
            goal: The goal dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        available_tools = {t["name"]: t for t in self.toolkit.get_tool_definitions()}
        
        # Check 1: All tools exist
        for step in plan:
            tool_name = step.get("tool")
            if tool_name and tool_name not in available_tools:
                issues.append(f"Step {step.get('step', '?')}: Unknown tool '{tool_name}'")
        
        # Check 2: Required parameters present
        for step in plan:
            tool_name = step.get("tool")
            if tool_name and tool_name in available_tools:
                tool_def = available_tools[tool_name]
                required_params = tool_def.get("input_schema", {}).get("required", [])
                step_input = step.get("input", {}) or {}
                
                for param in required_params:
                    if param not in step_input or step_input[param] is None:
                        issues.append(f"Step {step.get('step', '?')}: Missing required parameter '{param}' for tool '{tool_name}'")
        
        # Check 3: At least one action step exists
        if not any(step.get("action") in ["use tool", "synthesize"] for step in plan):
            issues.append("Plan has no actionable steps")
        
        # Check 4: Plan addresses goal
        tool_names = [step.get("tool") for step in plan if step.get("tool")]
        if goal.get("needs_order_info") and "get_order" not in tool_names:
            # Check if order_id is available for get_order
            if goal.get("order_id"):
                issues.append("Goal needs order info but plan doesn't call get_order")
        if goal.get("needs_tracking") and "track_shipment" not in tool_names:
            # Check if tracking_number is available for track_shipment
            if goal.get("tracking_number"):
                issues.append("Goal needs tracking but plan doesn't call track_shipment")
        if goal.get("needs_refund") and "process_refund" not in tool_names:
            if goal.get("order_id"):
                issues.append("Goal needs refund but plan doesn't call process_refund")
        
        return len(issues) == 0, issues
    
    def _is_complex_goal(self, goal: Dict[str, Any]) -> bool:
        """
        Determine if goal requires complex planning.
        
        Args:
            goal: The goal dictionary
            
        Returns:
            True if goal is complex and needs LLM planning, False otherwise
        """
        complexity_indicators = [
            # Multiple needs (order info + tracking + refund)
            len([k for k, v in goal.items() if k.startswith("needs_") and v]) > 1,
            # Refunds often need approval/validation
            goal.get("needs_refund", False),
            # Missing identifiers but still has needs
            (goal.get("needs_order_info") or goal.get("needs_tracking")) and 
            not goal.get("order_id") and not goal.get("tracking_number"),
            # Complex request mentioned
            "complex" in goal.get("core_request", "").lower(),
        ]
        # If 2+ indicators, it's complex
        return sum(complexity_indicators) >= 2
    
    def process_request(self, customer_message: str) -> str:
        """
        Main entry point for processing customer requests.
        Uses state machine and planning phase.
        
        Args:
            customer_message: The customer's message/request
            
        Returns:
            Final response string from the agent
        """
        print(f"CUSTOMER: {customer_message}")
        
        # Reset execution context
        self.execution_context = {
            "goal": None,
            "plan": None,
            "results": {},
            "attempts": 0
        }
        
        # Phase 1: Understand & Extract Goal
        self.transition_to(AgentState.UNDERSTANDING)
        goal = self._extract_goal_with_llm(customer_message)
        self.execution_context["goal"] = goal
        print(f"🎯 Goal: {json.dumps(goal, indent=2)}\n")
        
        # Phase 2: Create Execution Plan
        if self._is_complex_goal(goal):
            self.transition_to(AgentState.PLANNING)
            plan = self._create_execution_plan(goal)
            
            # Validate plan before execution
            is_valid, issues = self._validate_plan(plan, goal)
            if not is_valid:
                print(f"⚠️  Plan validation failed: {issues}")
                print(f"   Falling back to simple plan")
                plan = self._create_simple_plan(goal)
            else:
                print(f"✅ Plan validated successfully")
        else:
            # Use simple deterministic plan for straightforward queries
            plan = self._create_simple_plan(goal)
            print(f"📋 Using simple plan (no LLM planning needed)")
        
        self.execution_context["plan"] = plan
        
        # Initialize memory with user message
        self.memory.clear()
        self.memory.add_user_message(customer_message)
        
        # Phase 3: Execute Plan (ReAct loop)
        # Verification happens inside _react_loop or _execute_plan
        self.transition_to(AgentState.EXECUTING)
        final_response = self._react_loop(goal, plan)
        
        # Trust that loop/plan execution has already verified completion
        self.transition_to(AgentState.COMPLETE)
        return final_response
    
    def _execute_plan(self, goal: Dict[str, Any], plan: List[Dict[str, Any]]) -> str:
        """
        Execute plan steps sequentially.
        
        Args:
            goal: The goal dictionary
            plan: The execution plan to follow
            
        Returns:
            Final response string
        """
        results = {}
        tools_used = []
        
        print(f"📋 Executing plan with {len(plan)} steps\n")
        
        for step in plan:
            step_num = step.get("step", "?")
            action = step.get("action", "")
            tool_name = step.get("tool")
            tool_input = step.get("input")
            purpose = step.get("purpose", "")
            
            print(f"📋 Executing plan step {step_num}: {purpose}")
            
            if action == "use tool" and tool_name:
                # Execute tool step
                if not tool_input:
                    print(f"   ⚠️  Step {step_num} has no input, skipping")
                    continue
                
                tool_result = self._execute_tool_with_retry(tool_name, tool_input)
                results[tool_name] = tool_result
                tools_used.append(tool_name)
                
                # Add to memory for context
                self.memory.add_user_message(
                    f"Tool {tool_name} executed with result: {json.dumps(tool_result)}"
                )
                
                print(f"   ✅ Step {step_num} completed")
                
            elif action == "synthesize":
                # Synthesize final response from gathered data
                print(f"   📝 Synthesizing final response from gathered information")
                final_response = self._synthesize_response(goal, results)
                self.execution_context["tools_used"] = tools_used
                self.execution_context["results"] = results
                
                # Verify the synthesized response
                is_fulfilled = self._verify_goal_fulfillment_with_llm(final_response, goal, tools_used)
                if is_fulfilled:
                    print(f"   ✅ Response verified as complete")
                    return final_response
                else:
                    print(f"   ⚠️  Synthesized response verification failed, falling back to ReAct loop")
                    # Fall back to ReAct loop if synthesis doesn't fulfill goal
                    return self._react_loop(goal, None, is_retry=True)
            else:
                print(f"   ⚠️  Unknown action '{action}' in step {step_num}, skipping")
        
        # If we reach here, plan didn't have a synthesize step
        # Fall back to ReAct loop
        print(f"⚠️  Plan completed but no synthesis step found, falling back to ReAct loop")
        return self._react_loop(goal, None, is_retry=True)
    
    def _react_loop(self, goal: Dict[str, Any], plan: Optional[List[Dict[str, Any]]] = None, max_iterations: int = 10, is_retry: bool = False) -> str:
        """
        Agent ReAct (Reasoning + Acting) loop.
        
        If a plan is provided, executes plan steps sequentially.
        Otherwise, uses dynamic ReAct pattern.
        
        Args:
            goal: Goal dictionary extracted from user message
            plan: Optional execution plan (if None, uses ReAct pattern)
            max_iterations: Maximum number of loop iterations
            is_retry: Whether this is a retry attempt after verification failure
            
        Returns:
            Final response string
        """
        # If plan exists, execute it sequentially
        if plan and not is_retry:
            return self._execute_plan(goal, plan)
        
        # Otherwise, use dynamic ReAct loop
        tools_used = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"🔄 Iteration {iteration}")
            
            # Get current conversation history
            conversation_history = self.memory.get_history()
            
            print(f"📡 LLM API CALL #{self.llm.get_api_call_count() + 1}")
            print(f"   Tools available: {len(self.toolkit.get_tool_definitions())}")
            
            # Call LLM with current state
            try:
                llm_response = self.llm.call_llm(
                    messages=conversation_history,
                    tools=self.toolkit.get_tool_definitions(),
                    system=self.system_message.get_prompt()
                )
            except Exception as e:
                print(f"   ⚠️  LLM API call failed: {str(e)}")
                return "I apologize, but I encountered an error. Please try again later."
            
            # Parse response
            parsed = self.llm.parse_response(llm_response)
            
            if parsed["type"] == "empty":
                print(f"   ⚠️  Empty response from API")
                continue
            
            print(f"   Response type: {parsed['type']}")
            
            # Handle tool use
            if parsed["type"] == "tool_use":
                tool_name = parsed["tool_name"]
                tool_input = parsed["tool_input"]
                tool_use_id = parsed["tool_use_id"]
                
                print(f"\n🔧 LLM decided to use tool: {tool_name}")
                print(f"   Tool use ID: {tool_use_id}")
                print(f"   Input: {json.dumps(tool_input, indent=2)}")
                
                # Execute tool via toolkit with retry
                tool_result = self._execute_tool_with_retry(tool_name, tool_input)
                print(f"   Result: {json.dumps(tool_result, indent=2)}\n")
                
                # Track tool usage
                tools_used.append(tool_name)
                self.execution_context["tools_used"] = tools_used
                
                # Add to memory
                self.memory.add_assistant_message(llm_response.content)
                self.memory.add_tool_result(tool_use_id, tool_result)
                
                # Continue loop to process tool result
                continue
            
            # Handle text response (may also contain tool_use blocks)
            elif parsed["type"] == "text":
                # Check for mixed tool_use blocks
                if parsed.get("tool_use_blocks"):
                    # Process tool_use blocks first
                    for tool_block in parsed["tool_use_blocks"]:
                        tool_name = tool_block.name
                        tool_input = tool_block.input
                        tool_use_id = tool_block.id
                        
                        print(f"\n🔧 LLM decided to use tool: {tool_name}")
                        print(f"   Tool use ID: {tool_use_id}")
                        print(f"   Input: {json.dumps(tool_input, indent=2)}")
                        
                        # Execute tool with retry
                        tool_result = self._execute_tool_with_retry(tool_name, tool_input)
                        print(f"   Result: {json.dumps(tool_result, indent=2)}\n")
                        
                        # Track and store
                        tools_used.append(tool_name)
                        self.execution_context["tools_used"] = tools_used
                        self.memory.add_assistant_message([tool_block])
                        self.memory.add_tool_result(tool_use_id, tool_result)
                    
                    # Continue loop to process tool results
                    continue
                
                # Pure text response
                text_response = parsed.get("text", "")
                
                if not text_response:
                    print(f"   ⚠️  No text content found in response")
                    continue
                
                # Check if goal is fulfilled using LLM-based verification
                is_fulfilled = self._verify_goal_fulfillment_with_llm(text_response, goal, tools_used)
                
                if is_fulfilled:
                    print(f"\n✅ Goal fulfilled - LLM generated complete response")
                    print(f"   Tools used: {tools_used}")
                    return text_response
                else:
                    # Goal not fulfilled - add response and prompt for completion
                    print(f"\n⚠️  Goal not fulfilled - text response appears to be acknowledgment or incomplete")
                    print(f"   Text: {text_response[:150]}...")
                    print(f"   Tools used so far: {tools_used}")
                    print(f"   Continuing loop to gather more information...\n")
                    
                    # Add text response to memory
                    text_blocks = [block for block in parsed["blocks"] if block.type == "text"]
                    self.memory.add_assistant_message(text_blocks)
                    
                    # Prompt for completion
                    self.memory.add_user_message(
                        "Please use the available tools to gather the necessary information and provide a complete answer. Do not give preliminary acknowledgments."
                    )
                    
                    # Continue loop
                    continue
            else:
                print(f"   ⚠️  Unexpected response type: {parsed['type']}")
                continue
        
        # Max iterations reached
        print(f"\n⚠️  Max iterations reached without fulfilling goal")
        print(f"   Tools used: {tools_used}")
        return "I apologize, but I couldn't complete your request. Please try again or contact support."
    
    def show_api_call_summary(self):
        """Show all LLM API calls that were made."""
        print(f"\n{'='*70}")
        print(f"LLM API CALL SUMMARY")
        print(f"{'='*70}")
        print(f"Total calls made: {self.llm.get_api_call_count()}")
        print(f"Model used: {self.llm.model}")
        print(f"Provider: {self.llm.provider}")
        print(f"\nIn a real system, each call would:")
        print(f"  • Send HTTP request to API endpoint")
        print(f"  • Cost money (per token)")
        print(f"  • Take ~1-3 seconds")
        print(f"  • Return actual LLM reasoning")
        print(f"{'='*70}\n")


# ============= DEMONSTRATION =============

def main():
    agent = ReActAgent()
    final_response = agent.process_request(
        "Where is my order ORD-12345? I need to know when it will arrive."
    )
    
    print(final_response)
    
    # Show summary
    agent.show_api_call_summary()
    


if __name__ == "__main__":
    main()
