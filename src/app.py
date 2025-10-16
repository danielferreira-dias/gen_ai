from database.evaluation import EvaluationStorage
from services.evaluation_service import LLMJudge, LightHeuristic
import streamlit as st
import asyncio
import logging
import time
from services.process_service import ProcessingService
from services.pii_service import AzureLanguageService
from services.agent_service import Agent
from services.chroma_service import ChromaService, EmbeddingModel
from database.storage import ConversationStorage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- SETUP ----------
st.set_page_config(page_title="PII-Aware Chatbot", page_icon="")

logger.info("Initializing services...")
azure_service = AzureLanguageService()
embedding_model = EmbeddingModel(model_id="sentence-transformers/all-MiniLM-L6-v2")
chroma_service = ChromaService(embedding_model=embedding_model)
agent = Agent(model_name="gpt-5-chat", chroma_service=chroma_service)
llm_judge = LLMJudge( model_name="gpt-5-chat", db_path="./db/evaluation.db")
light_heuristic = LightHeuristic()
processing_Service = ProcessingService()
logger.info("Services initialized successfully")


## Storage
logger.info("Initializing Storages...")
storage = ConversationStorage()
evaluation_storage = EvaluationStorage()
logger.info("Storages initialized successfully")

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "message_metadata" not in st.session_state:
    st.session_state.message_metadata = {}  # Stores {message_index: {"message_id": int, "evaluation_id": int}}

# ---------- TITLE ----------
st.title("🤖 Agent Chatbot")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Settings")
    debug_mode = st.checkbox("Enable Debug Tracing", value=False)

    st.divider()

    st.header("Conversation History")

    if st.button("New Conversation"):
        logger.info("Starting new conversation - clearing session state")
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.session_state.message_metadata = {}
        st.rerun()

    st.divider()

    # PII Audit Log
    st.header("PII Audit")
    if st.button("View PII Log"):
        pii_logs = storage.get_pii_audit_log(conversation_id=st.session_state.conversation_id)
        if pii_logs:
            st.write(f"**PII Events in Current Chat:** {len(pii_logs)}")
            for log in pii_logs:
                with st.expander(f"Event {log['id']} - {log['timestamp'][:19]}"):
                    st.write(f"**Entities Detected:** {log['entities_detected']}")
                    st.json(log['entity_details'])
        else:
            st.info("No PII detected in this conversation")

# ---------- DISPLAY CHAT ----------
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Add feedback widget for assistant messages
        if msg["role"] == "assistant" and idx in st.session_state.message_metadata:
            metadata = st.session_state.message_metadata[idx]
            message_id = metadata.get("message_id")
            evaluation_id = metadata.get("evaluation_id")

            def handle_feedback():
                feedback_key = f"feedback_{idx}"
                if feedback_key in st.session_state and st.session_state[feedback_key] is not None:
                    feedback_value = st.session_state[feedback_key]
                    try:
                        evaluation_storage.save_user_feedback(
                            conversation_id=st.session_state.conversation_id,
                            message_id=message_id,
                            feedback=feedback_value,
                            evaluation_id=evaluation_id
                        )
                        logger.info(f"User feedback saved: message_id={message_id}, feedback={feedback_value}")
                        st.toast("Thanks for your feedback!" if feedback_value == 1 else "Thanks for letting us know!", icon="👍" if feedback_value == 1 else "👎")
                    except Exception as e:
                        logger.error(f"Error saving feedback: {e}")

            st.feedback(
                "thumbs",
                key=f"feedback_{idx}",
                on_change=handle_feedback
            )

# ---------- USER INPUT ----------
if user_input := st.chat_input("Type your message..."):
    logger.info(f"Received user input: {user_input[:50]}...")
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Store user message (will update with PII data later if detected)
    user_message_data = None

    # Create conversation on first message
    if st.session_state.conversation_id is None:
        st.session_state.conversation_id = storage.create_conversation()
        logger.info(f"Created new conversation with ID: {st.session_state.conversation_id}")

    # Create a placeholder for tracing
    with st.spinner("Processing..."):
        try:
            # Detect PII and tokenize
            with st.expander("🔍 PII Detection & Tokenization", expanded=debug_mode):
                st.write("**Step 1:** Detecting PII entities...")
                data = azure_service.pii_recognition_example(user_query=user_input)

                if data:
                    st.write("**Step 2:** PII Detection Results:")
                    st.json({
                        "entities_found": len(data.get('entities', [])),
                        "entities": [
                            {
                                "text": e.get('text'),
                                "category": e.get('category'),
                                "confidence": e.get('confidence_score')
                            } for e in data.get('entities', [])
                        ]
                    })

                st.write("**Step 3:** Tokenizing PII...")
                # Get existing conversation token map to avoid token collisions
                existing_token_map = storage.get_conversation_token_map(
                    conversation_id=st.session_state.conversation_id
                ) if st.session_state.conversation_id else {}

                process_data = processing_Service.tokenize_pii(
                    data=data,
                    existing_token_map=existing_token_map
                )
                tokenized_text = process_data.get('tokenized_text', user_input)

                if process_data.get('has_pii'):
                    st.success("✅ PII detected and tokenized")
                    st.write("**Tokenized Text:**")
                    st.code(tokenized_text)
                    st.write("**Token Map:**")
                    st.json(process_data.get('token_map', {}))
                else:
                    st.info("ℹ️ No PII detected - using original text")

            # Store user message with PII data
            logger.info(f"Storing user message in database (conversation_id: {st.session_state.conversation_id})")
            user_message_data = storage.add_message(
                conversation_id=st.session_state.conversation_id,
                role="user",
                content=user_input,
                tokenized_content=tokenized_text if process_data.get('has_pii') else None,
                has_pii=process_data.get('has_pii', False),
                pii_data=process_data if process_data.get('has_pii') else None
            )
            logger.info(f"User message stored successfully (message_id: {user_message_data})")

            # Update conversation-level token map if PII was detected
            if process_data.get('has_pii') and process_data.get('token_map'):
                logger.info("Updating conversation-level token map with new PII tokens")
                storage.update_conversation_token_map(
                    conversation_id=st.session_state.conversation_id,
                    new_token_map=process_data.get('token_map')
                )

        except Exception as e:
            logger.error(f"Error during PII processing: {e}", exc_info=True)
            st.error(f"⚠️ Error during PII processing: {e}")
            tokenized_text = user_input
            process_data = {'has_pii': False}

    # ---------- AGENT RESPONSE ----------
    with st.spinner("Agent is thinking..."):
        # Start timing for heuristics
        start_time = time.time()

        # Fetch conversation history from database (excluding current message)
        # This retrieves messages with tokenized content if PII was detected
        logger.info(f"Fetching conversation context (conversation_id: {st.session_state.conversation_id})")
        conversation_context = storage.get_conversation_context(
            conversation_id=st.session_state.conversation_id,
            limit=20
        )
        logger.info(f"Retrieved {len(conversation_context) if conversation_context else 0} messages from history")

        # Pass conversation history to agent
        logger.info(f"Invoking agent with query: {tokenized_text[:50]}...")
        agent_response = asyncio.run(agent.llm_response(
            user_query=tokenized_text,
            conversation_history=conversation_context
        ))

        # Extract response and tool calls
        response = agent_response.response
        tool_calls = agent_response.tool_calls

        # Calculate response time in milliseconds
        response_time_ms = (time.time() - start_time) * 1000

        logger.info(f"Agent response received: {response[:100]}...")
        logger.info(f"Tool invocations: {len(tool_calls)}")
        logger.info(f"Response time: {response_time_ms:.2f}ms")

    # Display tool invocations if any
    if tool_calls:
        with st.expander("🛠️ Tool Invocations & Vector Search Results", expanded=debug_mode):
            for idx, tool_call in enumerate(tool_calls, 1):
                st.write(f"**Tool {idx}: {tool_call['tool_name']}**")
                st.write(f"Query: `{tool_call['query']}`")
                st.write(f"Results Found: {tool_call['results_count']}")

                if tool_call['results']:
                    st.write("**Retrieved Documents:**")
                    for i, result in enumerate(tool_call['results'], 1):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Result {i}**")
                            with col2:
                                st.metric("Similarity Score", f"{result['Score']:.4f}")

                            st.markdown(f"*Source:* `{result['Source']}`")
                            st.text_area(
                                label=f"Content",
                                value=result['Content'][:400] + ("..." if len(result['Content']) > 400 else ""),
                                height=100,
                                key=f"tool_result_{idx}_{i}",
                                disabled=True
                            )
                if idx < len(tool_calls):
                    st.divider()

    # Store the tokenized assistant response BEFORE de-tokenizing
    # This ensures the agent always sees tokens in conversation history
    tokenized_response = response  # Keep the tokenized version

    # De-tokenize using conversation-level token map for display to user
    detokenized_response = response  # Default to original if no de-tokenization needed
    try:
        # Get conversation-level token map (accumulated across all messages)
        conversation_token_map = storage.get_conversation_token_map(
            conversation_id=st.session_state.conversation_id
        )

        if conversation_token_map:
            logger.info("Starting de-tokenization of agent response using conversation-level token map...")
            with st.expander("🔓 De-tokenization", expanded=debug_mode):
                st.write("**Agent Response (Tokenized):**")
                st.code(response)

                st.write("**Available Token Map:**")
                st.json(conversation_token_map)

                detokenized_response = processing_Service.detokenize_pii(
                    text=response,
                    token_map=conversation_token_map
                )

                st.write("**Agent Response (De-tokenized):**")
                st.code(detokenized_response)
                st.success("✅ PII restored in response")
                logger.info("De-tokenization complete")
    except Exception as e:
        logger.warning(f"De-tokenization skipped: {e}", exc_info=True)
        st.warning(f"⚠️ De-tokenization skipped: {e}")

    # Store assistant message with BOTH versions:
    # - content: de-tokenized (for display to user)
    # - tokenized_content: tokenized (for agent context)
    # - has_pii: True if response contains any tokens
    has_tokens_in_response = tokenized_response != detokenized_response

    logger.info("Storing assistant message in database...")
    assistant_message_id = storage.add_message(
        conversation_id=st.session_state.conversation_id,
        role="assistant",
        content=detokenized_response,  # Store de-tokenized for user display
        tokenized_content=tokenized_response if has_tokens_in_response else None,  # Store tokenized for agent context
        has_pii=has_tokens_in_response  # Mark as having PII if we de-tokenized it
    )
    logger.info(f"Assistant message stored successfully (message_id: {assistant_message_id})")

    # ---------- LIGHTWEIGHT HEURISTICS ----------
    try:
        # Prepare retrieved documents data from tool calls
        retrieved_docs = []
        if tool_calls:
            for tool_call in tool_calls:
                if tool_call.get('results'):
                    for result in tool_call['results']:
                        retrieved_docs.append({
                            'content': result.get('Content', ''),
                            'score': result.get('Score', 0.0),
                            'source': result.get('Source', '')
                        })

        # Calculate lightweight heuristics
        heuristics = light_heuristic.calculate_heuristics(
            agent_response=detokenized_response,
            retrieved_docs=retrieved_docs,
            response_time_ms=response_time_ms
        )

        logger.info(f"Lightweight heuristics calculated: {heuristics}")

        # Display heuristics in debug mode
        if debug_mode:
            with st.expander("📊 Lightweight Heuristics", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Response Length", heuristics['response_length'])
                    st.metric("Has Citation", "✅" if heuristics['has_citation'] else "❌")
                with col2:
                    st.metric("Retrieval Confidence", f"{heuristics['retrieval_confidence']:.2f}")
                    st.metric("Response Time", f"{heuristics['response_time_ms']:.0f} ms")
                with col3:
                    st.metric("Docs Retrieved", heuristics['num_docs_retrieved'])

    except Exception as e:
        logger.error(f"Error calculating lightweight heuristics: {e}", exc_info=True)
        heuristics = {}

    # ---------- EVALUATE ANSWER - LLM AS A JUDGE ----------
    # Evaluate AFTER de-tokenization so the judge sees the actual response
    evaluation_id = None
    with st.spinner("Evaluating response..."):
        try:
            judge_evaluation = asyncio.run(llm_judge.evaluate( user_query=user_input, agent_response=detokenized_response, conversation_id=st.session_state.conversation_id, message_id=assistant_message_id  # Link to assistant's message, not user's
            ))
            evaluation_id = llm_judge.storage.db_path 
            logger.info(f"LLM Judge evaluation complete: score={judge_evaluation.overall_score}/5")
            if debug_mode:
                with st.expander("🧑‍⚖️ LLM Judge Evaluation", expanded=True):
                    st.write(f"**Relevance:** {judge_evaluation.relevance}")
                    st.write(f"**Accuracy:** {judge_evaluation.accuracy}")
                    st.write(f"**PII Violation:** {judge_evaluation.pii_violation}")
                    st.write(f"**Safety Violation:** {judge_evaluation.safety_violation}")
                    st.write(f"**Clarity:** {judge_evaluation.clarity}")
                    st.write(f"**Overall Score:** {judge_evaluation.overall_score}/5")
                    st.write("**Rationale:**")
                    st.markdown(judge_evaluation.rationale)
        except Exception as e:
            logger.error(f"Error during LLM Judge evaluation: {e}", exc_info=True)
            st.error(f"⚠️ Error during evaluation: {e}")

    # Store message metadata for feedback widget
    message_index = len(st.session_state.messages)
    st.session_state.message_metadata[message_index] = {
        "message_id": assistant_message_id,
        "evaluation_id": evaluation_id
    }

    st.session_state.messages.append({"role": "assistant", "content": detokenized_response})
    with st.chat_message("assistant"):
        st.markdown(detokenized_response)

        # Add feedback widget for the newly added message
        def handle_new_feedback():
            feedback_key = f"feedback_{message_index}"
            if feedback_key in st.session_state and st.session_state[feedback_key] is not None:
                feedback_value = st.session_state[feedback_key]
                try:
                    evaluation_storage.save_user_feedback(
                        conversation_id=st.session_state.conversation_id,
                        message_id=assistant_message_id,
                        feedback=feedback_value,
                        evaluation_id=evaluation_id
                    )
                    logger.info(f"User feedback saved: message_id={assistant_message_id}, feedback={feedback_value}")
                    st.toast("Thanks for your feedback!" if feedback_value == 1 else "Thanks for letting us know!", icon="👍" if feedback_value == 1 else "👎")
                except Exception as e:
                    logger.error(f"Error saving feedback: {e}")

        st.feedback(
            "thumbs",
            key=f"feedback_{message_index}",
            on_change=handle_new_feedback
        )

    logger.info("Message exchange complete")
