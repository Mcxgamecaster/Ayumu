import os
import discord
from discord.ext import commands
from google import genai
from google.genai import types
import json
import asyncio
from collections import deque
import io
from datetime import datetime, timezone
import aiohttp
import tempfile
import mimetypes
import base64
import re


# --- Load Settings ---
config = {}
try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
except FileNotFoundError:
    print("config.json not found! Using environment variables and defaults for some settings.")
    # If config.json is optional or primarily for non-secret settings, handle its absence.

# --- Essential Variables ---
# Try to get from environment variables first, then fallback to config.json (for non-secret local testing if needed)
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN') # Only read from env var for secret
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') # Only read from env var for secret

# --- Default Settings (can still be loaded from config.json) ---
DEFAULT_CONTEXT_SIZE = config.get('default_context_size', 100000)
DEFAULT_MAX_OUTPUT_TOKENS = config.get('default_max_output_tokens', 65536)
DEFAULT_MESSAGE_LIMIT = config.get('default_message_limit', 35)
DEFAULT_MODEL = config.get('default_model', 'gemini-2.5-pro')
DEFAULT_INCLUDE_THOUGHTS = config.get('default_include_thoughts', False)
DEFAULT_THINKING_BUDGET = config.get('default_thinking_budget', 0)
DEFAULT_PROCESS_IMAGES = config.get('default_process_images', True)
DEFAULT_PROCESS_VIDEOS = config.get('default_process_videos', True)

# --- Prerequisite Checks ---
if not DISCORD_TOKEN:
    print("Discord Token is missing! Please set the DISCORD_TOKEN environment variable.")
    exit()
if not GOOGLE_API_KEY:
    print("Google API Key is missing! Please set the GOOGLE_API_KEY environment variable.")
    exit()

try:
    with open('system_prompt.txt', 'r', encoding='utf-8') as f:
        system_instructions = f.read()
except FileNotFoundError:
    system_instructions = "You are a helpful assistant."

try:
    with open('personality_profile.txt', 'r', encoding='utf-8') as f:
        personality_profile = f.read()
except FileNotFoundError:
    personality_profile = ""

# --- Configure Google AI Client ---
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

# --- Configure Discord Bot ---
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.dm_messages = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- Bot State Management ---
channel_settings = {}
server_settings = {}
chat_history = {}
channel_locks = {}
last_responses = {}

# --- Multimodal Models List ---
MULTIMODAL_MODELS = [
    "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-2.5-pro-preview-06-05", "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-lite-preview-06-17"
]

# YouTube URL regex pattern
YOUTUBE_URL_PATTERN = re.compile(
    r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|v/)|youtu\.be/)([a-zA-Z0-9_-]+)'
)

# Modal class for editing prompts and personality
class EditPromptsModal(discord.ui.Modal):
    def __init__(self):
        super().__init__(title="แก้ไข Prompt และ Personality", timeout=None)

        self.system_prompt_input = discord.ui.TextInput(
            label="System Prompt (สำหรับผู้ใช้ขั้นสูง)",
            style=discord.TextStyle.paragraph,
            placeholder="เว้นว่างเพื่อใช้ค่าเริ่มต้นจากไฟล์",
            required=False,
            max_length=2000,
        )

        self.personality_input = discord.ui.TextInput(
            label="Personality Profile",
            style=discord.TextStyle.paragraph,
            placeholder="เว้นว่างเพื่อใช้ค่าเริ่มต้นจากไฟล์",
            required=False,
            max_length=1500,
        )

        self.add_item(self.system_prompt_input)
        self.add_item(self.personality_input)

    async def on_submit(self, interaction: discord.Interaction):
        storage_key = interaction.user.id if interaction.guild is None else interaction.guild.id
        
        new_system_prompt = self.system_prompt_input.value
        new_personality = self.personality_input.value
        
        feedback_messages = []

        if not new_system_prompt.strip():
            if server_settings.get(storage_key, {}).pop("custom_system_prompt", None) is not None:
                feedback_messages.append("✅ System Prompt ถูกรีเซ็ตกลับไปเป็นค่าเริ่มต้นแล้ว")
        else:
            server_settings.setdefault(storage_key, {})["custom_system_prompt"] = new_system_prompt
            feedback_messages.append("✅ System Prompt ถูกอัปเดตแล้ว")

        if not new_personality.strip():
            if server_settings.get(storage_key, {}).pop("custom_personality", None) is not None:
                feedback_messages.append("✅ Personality ถูกรีเซ็ตกลับไปเป็นค่าเริ่มต้นแล้ว")
        else:
            server_settings.setdefault(storage_key, {})["custom_personality"] = new_personality
            feedback_messages.append("✅ Personality ถูกอัปเดตแล้ว")
        
        if not feedback_messages:
            final_response = "ℹ️ ไม่มีการเปลี่ยนแปลงค่าใดๆ"
        else:
            final_response = "\n".join(feedback_messages)

        await interaction.response.send_message(final_response, ephemeral=True)


def _extract_youtube_urls(text: str) -> list:
    """Extract YouTube URLs from text"""
    if not text:
        return []
    
    matches = YOUTUBE_URL_PATTERN.findall(text)
    youtube_urls = []
    
    for match in matches:
        # Reconstruct the full YouTube URL
        youtube_url = f"https://www.youtube.com/watch?v={match}"
        youtube_urls.append(youtube_url)
    
    return youtube_urls


async def _process_attachments(message: discord.Message, settings: dict) -> list:
    """
    Processes attachments in a message, downloading them and preparing them
    for the Gemini API using the new genai client. Checks settings to decide whether to process.
    Also processes YouTube URLs from message content.

    Returns a list of parts (image data as types.Part objects).
    """
    selected_model = settings.get("model", DEFAULT_MODEL)
    if not any(model_name in selected_model for model_name in MULTIMODAL_MODELS):
        print(f"Skipping video processing: Model '{selected_model}' doesn't support video.")
        return []

    processed_parts = []
    
    # Get file processing settings (default to True if not set)
    process_images_enabled = settings.get("process_images", True)
    process_videos_enabled = settings.get("process_videos", True)

    # Process YouTube URLs from message content
    if process_videos_enabled and message.content:
        youtube_urls = _extract_youtube_urls(message.content)
        for youtube_url in youtube_urls:
            print(f"Found YouTube URL: {youtube_url}")
            try:
                # Create FileData part for YouTube URL
                youtube_part = types.Part.from_uri(
                    file_uri=youtube_url,
                    mime_type="video/mp4"  # YouTube videos are typically MP4
                )
                processed_parts.append(youtube_part)
                print(f"   + Successfully added YouTube video: {youtube_url}")
            except Exception as e:
                print(f"   - Error processing YouTube URL {youtube_url}: {e}")

    # Process Discord attachments
    if not message.attachments:
        return processed_parts

    async with aiohttp.ClientSession() as session:
        for attachment in message.attachments:
            print(f"Found attachment: {attachment.filename} ({attachment.content_type})")

            # --- Image Handling (Inline) ---
            if attachment.content_type and attachment.content_type.startswith("image/"):
                if not process_images_enabled:
                    print(f"-> Skipping image '{attachment.filename}' because processing is disabled.")
                    continue

                print(f"-> Processing image '{attachment.filename}'...")
                try:
                    async with session.get(attachment.url) as resp:
                        if resp.status == 200:
                            image_bytes = await resp.read()
                            # Create the image part correctly
                            image_part = types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=attachment.content_type
                            )
                            processed_parts.append(image_part)
                            print(f"   + Successfully processed image: {attachment.filename}")
                        else:
                            print(f"   - Failed to download image {attachment.filename}: HTTP {resp.status}")
                except Exception as e:
                    print(f"   - Error processing image {attachment.filename}: {e}")

            # --- Video Handling (using File API) ---
            elif attachment.content_type and attachment.content_type.startswith("video/"):
                if not process_videos_enabled:
                    print(f"-> Skipping video '{attachment.filename}' because processing is disabled.")
                    continue

                print(f"-> Processing video '{attachment.filename}' via File API...")
                temp_file_path = None
                try:

                        # Use File API for larger files
                        print(f"   + Using File API for large video '{attachment.filename}'...")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{attachment.filename}") as temp_file:
                            temp_file_path = temp_file.name
                            async with session.get(attachment.url) as resp:
                                if resp.status == 200:
                                    temp_file.write(await resp.read())
                                    print(f"   + Downloaded video '{attachment.filename}' to temporary file.")
                                else:
                                    print(f"   - Failed to download video {attachment.filename}: HTTP {resp.status}")
                                    continue

                        print(f"   + Uploading '{attachment.filename}' to Google File API...")
                        loop = asyncio.get_running_loop()
                        
                        # Upload file using new genai client - FIXED: Use 'file' parameter instead of 'path'
                        google_file = await loop.run_in_executor(
                            None,
                            lambda: genai_client.files.upload(file=temp_file_path)
                        )

                        # Wait for processing
                        while google_file.state == "PROCESSING":
                            print(f"   ... Waiting for '{attachment.filename}' to be processed...")
                            await asyncio.sleep(10)
                            google_file = await loop.run_in_executor(
                                None,
                                lambda: genai_client.files.get(name=google_file.name)
                            )

                        if google_file.state == "ACTIVE":
                            video_part = types.Part.from_uri(
                                file_uri=google_file.uri,
                                mime_type=google_file.mime_type
                            )
                            processed_parts.append(video_part)
                            print(f"   + Successfully uploaded and processed video: {attachment.filename}")
                        else:
                            print(f"   - Google File API failed to process video '{attachment.filename}'. State: {google_file.state}")
                            await message.channel.send(f"Sorry, I couldn't process the video '{attachment.filename}'. Please try again.")

                except Exception as e:
                    print(f"   - An error occurred while processing video {attachment.filename}: {e}")
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        print(f"   + Removed temporary file: {temp_file_path}")

    return processed_parts

# Updated function to properly handle parts conversion for history storage
def _convert_parts_to_dict(parts):
    """Convert types.Part objects to dictionary format for storage"""
    parts_dict = []
    for part in parts:
        if hasattr(part, 'text') and part.text:
            parts_dict.append({"text": part.text})
        elif hasattr(part, 'inline_data') and part.inline_data:
            # For inline data, we need to store the base64 encoded data
            import base64
            encoded_data = base64.b64encode(part.inline_data.data).decode('utf-8')
            parts_dict.append({
                "inline_data": {
                    "mime_type": part.inline_data.mime_type,
                    "data": encoded_data
                }
            })
        elif hasattr(part, 'file_data') and part.file_data:
            parts_dict.append({
                "file_data": {
                    "mime_type": part.file_data.mime_type,
                    "file_uri": part.file_data.file_uri
                }
            })
    return parts_dict

# Updated function to convert dictionary format back to types.Part objects
def _convert_dict_to_parts(parts_dict):
    """Convert dictionary format back to types.Part objects"""
    parts = []
    for part_dict in parts_dict:
        if "text" in part_dict:
            parts.append(types.Part.from_text(text=part_dict["text"]))
        elif "inline_data" in part_dict:
            import base64
            # Decode the base64 data back to bytes
            data_bytes = base64.b64decode(part_dict["inline_data"]["data"])
            parts.append(types.Part.from_bytes(
                data=data_bytes,
                mime_type=part_dict["inline_data"]["mime_type"]
            ))
        elif "file_data" in part_dict:
            parts.append(types.Part.from_uri(
                file_uri=part_dict["file_data"]["file_uri"],
                mime_type=part_dict["file_data"]["mime_type"]
            ))
    return parts

# Updated _prepare_request_payload function
def _prepare_request_payload(storage_key, channel_id):
    """
    Prepares all data to be sent to Google AI API using the new genai client format.
    This function doesn't make the actual API call.
    """
    default_settings_dict = {
        "context_size": DEFAULT_CONTEXT_SIZE,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "message_limit": DEFAULT_MESSAGE_LIMIT,
        "model": DEFAULT_MODEL,
        "include_thoughts": DEFAULT_INCLUDE_THOUGHTS,
        "thinking_budget": DEFAULT_THINKING_BUDGET
    }
    
    user_specific_settings = server_settings.get(storage_key, {})
    settings = default_settings_dict.copy()
    settings.update(user_specific_settings)

    history = chat_history.setdefault(storage_key, {}).setdefault(channel_id, deque(maxlen=settings["context_size"]))
    limited_history = list(history)[-settings["message_limit"]:]

    # --- Dynamic System Prompt Generation ---
    active_system_prompt = user_specific_settings.get("custom_system_prompt", system_instructions)
    active_personality = user_specific_settings.get("custom_personality", personality_profile)
    
    # Combine prompts, checking if personality is empty
    if active_personality and active_personality.strip():
        final_system_prompt = f"{active_system_prompt}\n\n{active_personality}"
    else:
        final_system_prompt = active_system_prompt

    # Create contents using new genai types
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=final_system_prompt)]
        ),
        types.Content(
            role="model",
            parts=[types.Part.from_text(text="รับทราบค่าาา! อายูมุพร้อมคุยกับทุกคนแล้วค่ะ! (｡•̀ᴗ-)✧")]
        )
    ]
    
    # Convert history to new format using helper function
    for msg in limited_history:
        role = msg["role"]
        parts = _convert_dict_to_parts(msg["parts"])
        contents.append(types.Content(role=role, parts=parts))
    
    # Create safety settings
    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE
        ),
    ]
    
    # Create generation config with safety settings included
    generation_config = types.GenerateContentConfig(
        temperature=1.0,
        top_p=1.0,
        max_output_tokens=settings["max_output_tokens"],
        response_mime_type="text/plain",
        safety_settings=safety_settings
    )

    # Add thinking config if enabled
    if settings["include_thoughts"] and settings["thinking_budget"] > 0:
        generation_config.thinking_config = types.ThinkingConfig(
            thinking_budget=settings['thinking_budget']
        )
    
    return contents, generation_config, settings["model"]

async def generate_response(storage_key, channel_id):
    history = chat_history.setdefault(storage_key, {}).setdefault(channel_id, deque(maxlen=DEFAULT_CONTEXT_SIZE))
    contents, generation_config, model_name = _prepare_request_payload(storage_key, channel_id)
    
    try:
        # Use the new genai client to generate content
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: genai_client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config
                # safety_settings removed from here - now in config
            )
        )
        
        last_responses.setdefault(storage_key, {})[channel_id] = response
        
        thought_text, reply_text = "", ""
        
        if response.candidates and len(response.candidates) > 0 and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'thought') and getattr(part, 'thought', False):
                    thought_text += part.text
                else:
                    reply_text += part.text
        
        if not reply_text.strip():
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                reply_text = f"อายูมุไม่สามารถตอบได้ค่ะ เนื่องจาก: {response.prompt_feedback.block_reason}"
            else:
                reply_text = "อายูมุคิดคำตอบไม่ออกเลยตอนนี้ >.<"
        
        # Add to history using new format
        history.append({
            "role": "model", 
            "parts": [{"text": reply_text}]
        })
        
        return {"thought": thought_text, "reply": reply_text}
        
    except Exception as e:
        print(f"Error calling Google AI API: {e}")
        if storage_key in last_responses and channel_id in last_responses[storage_key]:
            del last_responses[storage_key][channel_id]
        return {"thought": "", "reply": f"ขออภัยค่ะ เกิดข้อผิดพลาดบางอย่าง ({e})"}
    
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    print('Ayumu is ready for commands!')
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    is_dm = isinstance(message.channel, discord.DMChannel)
    storage_key = message.author.id if is_dm else message.guild.id
    channel_id = message.channel.id
    was_pinged = bot.user in message.mentions
    is_reply_to_bot = (message.reference and message.reference.cached_message and message.reference.cached_message.author == bot.user)
    is_priority_override = was_pinged or is_reply_to_bot
    
    lock = channel_locks.setdefault(channel_id, asyncio.Lock())
    if lock.locked() and not is_priority_override:
        print(f"[{channel_id}] Dropping message (not priority)...")
        return
    
    is_enabled_channel = False
    if not is_dm:
        is_enabled_channel = channel_settings.get(storage_key, {}).get(channel_id, {}).get("enabled", False)

    if is_dm or is_enabled_channel:
        user_specific_settings = server_settings.get(storage_key, {})
        current_settings = {
            "model": user_specific_settings.get("model", DEFAULT_MODEL),
            "process_images": user_specific_settings.get("process_images", DEFAULT_PROCESS_IMAGES),
            "process_videos": user_specific_settings.get("process_videos", DEFAULT_PROCESS_VIDEOS)
        }

        user_parts = []
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        attachment_parts = await _process_attachments(message, current_settings)
        
        text_content = ""
        if message.reference and message.reference.message_id:
            try:
                ref_msg = await message.channel.fetch_message(message.reference.message_id)
                orig_author, orig_content = ref_msg.author.display_name, ref_msg.content
                if ref_msg.author.id == bot.user.id:
                    lines = orig_content.split('\n')
                    content_lines = [line for line in lines if not line.strip().startswith('>')]
                    orig_content = '\n'.join(content_lines).strip()
                text_content = f'[{now_utc}] User "{message.author.display_name}" (ID: {message.author.id}): [ตอบกลับถึง "{orig_author}": "{orig_content}"] >> {message.content}'
            except discord.NotFound:
                text_content = f'[{now_utc}] User "{message.author.display_name}" (ID: {message.author.id}): [ตอบกลับข้อความที่ถูกลบไปแล้ว] >> {message.content}'
            except Exception as e:
                print(f"Error fetching referenced message: {e}")
                text_content = f'[{now_utc}] User "{message.author.display_name}" (ID: {message.author.id}): {message.content}'
        else:
            text_content = f'[{now_utc}] User "{message.author.display_name}" (ID: {message.author.id}): {message.content}'
        
        if text_content.strip():
            user_parts.append(types.Part.from_text(text=text_content))
        
        user_parts.extend(attachment_parts)
        
        if user_parts:
            history = chat_history.setdefault(storage_key, {}).setdefault(channel_id, deque(maxlen=DEFAULT_CONTEXT_SIZE))
            # Convert parts to dict format for storage
            parts_dict = _convert_parts_to_dict(user_parts)    
            history.append({"role": "user", "parts": parts_dict})

    should_reply = False
    if is_dm:
        should_reply = True
    elif is_enabled_channel:
        channel_conf = channel_settings.get(storage_key, {}).get(channel_id, {})
        is_autoreply_on = channel_conf.get("autoreply", False)
        # Check for YouTube URLs in the message content
        has_youtube_url = bool(_extract_youtube_urls(message.content)) if message.content else False
        if is_autoreply_on or is_priority_override or message.attachments or has_youtube_url:
            should_reply = True

    if should_reply:
        history_for_channel = chat_history.get(storage_key, {}).get(channel_id, [])
        if not history_for_channel:
            return
        
        async with lock:
            print(f"[{channel_id}] Locked channel to generate response...")
            async with message.channel.typing():
                response_data = await generate_response(storage_key, channel_id)
                if response_data and response_data.get("reply"):
                    final_message = ""
                    thought_text = response_data.get("thought", "").strip()
                    reply_text = response_data.get("reply", "")
                    
                    if thought_text:
                        formatted_thought = "\n".join([f"> {line}" for line in thought_text.split('\n')])
                        final_message += f"{formatted_thought}\n"
                    
                    final_message += reply_text
                    await message.reply(final_message, mention_author=False)
            print(f"[{channel_id}] Sent response and unlocked channel.")
    
    await bot.process_commands(message)

# --- Utility function for slash commands ---
def get_storage_key(interaction: discord.Interaction) -> int:
    return interaction.user.id if interaction.guild is None else interaction.guild.id

# (Commands from enablellm to deletemessage remain unchanged)
@bot.tree.command(name="enablellm", description="เปิดใช้งาน AI ในช่องนี้ (ใช้ในเซิร์ฟเวอร์เท่านั้น)")
async def enablellm(interaction: discord.Interaction):
    if interaction.guild is None:
        await interaction.response.send_message("คำสั่งนี้ใช้ได้เฉพาะในเซิร์ฟเวอร์เท่านั้นค่ะ", ephemeral=True)
        return
    guild_id = interaction.guild.id; channel_id = interaction.channel.id
    channel_settings.setdefault(guild_id, {}).setdefault(channel_id, {})["enabled"] = True
    await interaction.response.send_message(f"เปิดใช้งาน AI ในช่อง <#{channel_id}> แล้วค่ะ!", ephemeral=True)

@bot.tree.command(name="disablellm", description="ปิดใช้งาน AI ในช่องนี้ (ใช้ในเซิร์ฟเวอร์เท่านั้น)")
async def disablellm(interaction: discord.Interaction):
    if interaction.guild is None:
        await interaction.response.send_message("คำสั่งนี้ใช้ได้เฉพาะในเซิร์ฟเวอร์เท่านั้นค่ะ", ephemeral=True)
        return
    guild_id = interaction.guild.id; channel_id = interaction.channel.id
    if guild_id in channel_settings and channel_id in channel_settings[guild_id]:
        channel_settings[guild_id][channel_id]["enabled"] = False
    await interaction.response.send_message(f"ปิดใช้งาน AI ในช่อง <#{channel_id}> แล้วค่ะ", ephemeral=True)

@bot.tree.command(name="autoreply", description="เปิด/ปิด การตอบกลับอัตโนมัติ (ใช้ในเซิร์ฟเวอร์เท่านั้น)")
@discord.app_commands.choices(status=[discord.app_commands.Choice(name="On", value="on"), discord.app_commands.Choice(name="Off", value="off")])
async def autoreply(interaction: discord.Interaction, status: discord.app_commands.Choice[str]):
    if interaction.guild is None:
        await interaction.response.send_message("คำสั่งนี้ใช้ได้เฉพาะในเซิร์ฟเวอร์เท่านั้นค่ะ", ephemeral=True)
        return
    guild_id = interaction.guild.id; channel_id = interaction.channel.id
    is_on = status.value == "on"; channel_settings.setdefault(guild_id, {}).setdefault(channel_id, {})["autoreply"] = is_on
    await interaction.response.send_message(f"{'เปิด' if is_on else 'ปิด'}การตอบกลับอัตโนมัติในช่องนี้แล้วค่ะ", ephemeral=True)

@bot.tree.command(name="ayumu", description="พูดคุยกับอายูมุ")
@discord.app_commands.describe(message="ข้อความที่ต้องการส่ง")
async def ayumu(interaction: discord.Interaction, message: str):
    is_dm = interaction.guild is None
    storage_key = get_storage_key(interaction)
    channel_id = interaction.channel.id
    if not is_dm and not channel_settings.get(storage_key, {}).get(channel_id, {}).get("enabled", False):
        await interaction.response.send_message("AI ยังไม่ได้ถูกเปิดใช้งานในช่องนี้ค่ะ ใช้ `/enablellm` ก่อนนะคะ", ephemeral=True)
        return
    await interaction.response.send_message(f"**คุณ:** {message}\n**น้อนอายูมุ:** กำลังคิด...", ephemeral=False)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    user_message_content = f'[{now_utc}] User "{interaction.user.display_name}" (ID: {interaction.user.id}): {message}'
    history = chat_history.setdefault(storage_key, {}).setdefault(channel_id, deque(maxlen=DEFAULT_CONTEXT_SIZE))
    history.append({"role": "user", "parts": [{"text": user_message_content}]})
    response_data = await generate_response(storage_key, channel_id)
    if response_data and response_data.get("reply"):
        thought_text, reply_text = response_data.get("thought", "").strip(), response_data.get("reply", "")
        final_content = f"**คุณ:** {message}\n"
        if thought_text:
            formatted_thought = "\n".join([f"> {line}" for line in thought_text.split('\n')])
            final_content += f"{formatted_thought}\n"
        final_content += f"**น้อนอายูมุ:** {reply_text}"
        await interaction.edit_original_response(content=final_content)
    else:
        await interaction.edit_original_response(content="ขออภัยค่ะ ไม่สามารถสร้างคำตอบได้ในขณะนี้")

@bot.tree.command(name="clearmemorychannel", description="ล้างประวัติการสนทนาของ AI ในช่อง/แชทนี้")
async def clearmemorychannel(interaction: discord.Interaction):
    storage_key, channel_id = get_storage_key(interaction), interaction.channel.id
    if storage_key in chat_history and channel_id in chat_history[storage_key]:
        chat_history[storage_key][channel_id].clear()
        await interaction.response.send_message("ล้างประวัติการสนทนาในแชทนี้เรียบร้อยค่ะ", ephemeral=True)
    else:
        await interaction.response.send_message("ยังไม่มีประวัติการสนทนาในแชทนี้ค่ะ", ephemeral=True)

@bot.tree.command(name="deletemessage", description="ลบข้อความล่าสุดที่บอทส่งในแชทนี้")
@discord.app_commands.describe(amount="จำนวนข้อความที่จะลบ (1-50)")
async def deletemessage(interaction: discord.Interaction, amount: discord.app_commands.Range[int, 1, 50]):
    await interaction.response.defer(ephemeral=True, thinking=True)
    channel = interaction.channel
    if interaction.guild:
        bot_member = interaction.guild.get_member(bot.user.id)
        if not channel.permissions_for(bot_member).manage_messages:
            await interaction.followup.send("อายูมุไม่มีสิทธิ์ `Manage Messages` ค่ะ", ephemeral=True); return
    messages_to_delete = []
    async for message in channel.history(limit=100):
        if message.author.id == bot.user.id:
            messages_to_delete.append(message)
            if len(messages_to_delete) >= amount: break
    if not messages_to_delete:
        await interaction.followup.send("ไม่พบข้อความล่าสุดของอายูมุที่จะลบค่ะ", ephemeral=True); return
    deleted_count = 0
    try:
        if interaction.guild and len(messages_to_delete) > 1:
            await channel.delete_messages(messages_to_delete); deleted_count = len(messages_to_delete)
        else:
            for msg in messages_to_delete: await msg.delete(); deleted_count += 1; await asyncio.sleep(0.5)
    except discord.HTTPException as e:
        print(f"Bulk delete failed: {e}")
        for msg in messages_to_delete:
            try: await msg.delete(); deleted_count += 1; await asyncio.sleep(0.5)
            except discord.HTTPException as individual_e: print(f"Could not delete message {msg.id}: {individual_e}")
    await interaction.followup.send(f"ลบข้อความล่าสุดของอายูมุไป {deleted_count} ข้อความค่ะ!", ephemeral=True)


@bot.tree.command(name="dumpcontext", description="ส่งออกประวัติการสนทนาปัจจุบัน (context) เป็นไฟล์ .txt")
async def dumpcontext(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    storage_key, channel_id = get_storage_key(interaction), interaction.channel.id
    if storage_key not in chat_history or channel_id not in chat_history.get(storage_key, {}):
        await interaction.followup.send("ไม่พบประวัติการสนทนาสำหรับช่อง/แชทนี้ค่ะ", ephemeral=True); return
    user_specific_settings = server_settings.get(storage_key, {})
    active_system_prompt = user_specific_settings.get("custom_system_prompt", system_instructions)
    active_personality = user_specific_settings.get("custom_personality", personality_profile)
    final_system_prompt = f"{active_system_prompt}\n\n{active_personality}"
    
    dump_content = []
    dump_content.append(f"Server ID: {storage_key}" if interaction.guild else f"User ID (DM): {storage_key}")
    dump_content.append(f"Channel ID: {channel_id}\n" + "="*50)
    dump_content.append("\n--- SYSTEM PROMPT (Role: User) ---\n" + final_system_prompt + "\n" + "="*50)
    dump_content.append("\n--- INITIAL RESPONSE (Role: Model) ---\nรับทราบค่าาา!...\n" + "="*50)
    dump_content.append("\n--- CHAT HISTORY (Oldest to Newest) ---")
    history_deque = chat_history[storage_key][channel_id]
    if not history_deque: dump_content.append("[History is empty]")
    else:
        for i, message in enumerate(history_deque):
            role = message.get('role', 'unknown').upper()
            parts_repr = []
            if 'parts' in message:
                for part in message['parts']:
                    if 'text' in part: parts_repr.append(f"[TEXT]: {part['text'][:200]}...")
                    elif 'inline_data' in part: parts_repr.append(f"[IMAGE]: {part['inline_data']['mime_type']}")
                    elif 'file_data' in part: parts_repr.append(f"[VIDEO]: {part['file_data']['file_uri']}")
            dump_content.append(f"\n[{i+1}] ROLE: {role}\n" + "\n".join(parts_repr) + "\n" + "-"*20)
    dump_content.append("\n--- END OF DUMP ---")
    final_string = "\n".join(dump_content)
    dump_file = discord.File(fp=io.BytesIO(final_string.encode('utf-8')), filename=f"context_dump_{channel_id}.txt")
    await interaction.followup.send("นี่คือไฟล์ข้อมูลประวัติการสนทนาทั้งหมดค่ะ", file=dump_file, ephemeral=True)

@bot.tree.command(name="synchistory", description="ซิงค์ประวัติล่าสุดในช่องเข้าสู่ความจำ (รองรับรูปภาพ/วิดีโอ)")
@discord.app_commands.describe(amount="จำนวนข้อความล่าสุดที่จะซิงค์ (1-200)")
@discord.app_commands.checks.cooldown(1, 60.0, key=lambda i: (i.guild_id or i.user.id, i.channel_id))
async def synchistory(interaction: discord.Interaction, amount: discord.app_commands.Range[int, 1, 200]):
    await interaction.response.defer(ephemeral=True, thinking=True)
    storage_key, channel_id = get_storage_key(interaction), interaction.channel.id
    history_deque = chat_history.setdefault(storage_key, {}).setdefault(channel_id, deque(maxlen=DEFAULT_CONTEXT_SIZE))
    history_deque.clear()
    await interaction.followup.send(f"กำลังดึง {amount} ข้อความล่าสุดและประมวลผลไฟล์แนบ (อาจใช้เวลานาน)...", ephemeral=True)
    messages = [msg async for msg in interaction.channel.history(limit=amount)]
    messages.reverse()
    # <<< --- [MODIFIED BLOCK START] --- >>>
    user_specific_settings = server_settings.get(storage_key, {})
    current_settings = {
        "model": user_specific_settings.get("model", DEFAULT_MODEL),
        "process_images": user_specific_settings.get("process_images", DEFAULT_PROCESS_IMAGES),
        "process_videos": user_specific_settings.get("process_videos", DEFAULT_PROCESS_VIDEOS)
    }
    # <<< --- [MODIFIED BLOCK END] --- >>>
    formatted_history = []
    processed_count = 0
    for message in messages:
        if not message.content and not message.attachments: continue
        parts = []
        role = "model" if message.author.id == bot.user.id else "user"
        if message.attachments:
            attachment_parts = await _process_attachments(message, current_settings)
            parts.extend(attachment_parts)
        text_content = ""
        if message.content:
            if role == "model":
                lines = message.content.split('\n')
                content_lines = [line for line in lines if not line.strip().startswith('>')]
                text_content = '\n'.join(content_lines).strip()
            else:
                timestamp = message.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                text_content = f'[{timestamp}] User "{message.author.display_name}" (ID: {message.author.id}): {message.content}'
        if text_content: parts.insert(0, {"text": text_content})
        if parts:
            formatted_history.append({"role": role, "parts": parts})
            processed_count += 1
    history_deque.extend(formatted_history)
    await interaction.edit_original_response(content=f"✅ ซิงค์และประมวลผล **{processed_count}** ข้อความ (รวมไฟล์แนบ) เข้าสู่ความจำเรียบร้อยแล้วค่ะ!")

# ... (reroll, personality, debug, and settings commands remain the same as your latest version)
@bot.tree.command(name="reroll", description="ลบคำตอบล่าสุดของอายูมุและสร้างคำตอบใหม่")
@discord.app_commands.checks.cooldown(1, 10.0, key=lambda i: (i.guild_id or i.user.id, i.channel_id))
async def reroll(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    storage_key, channel_id = get_storage_key(interaction), interaction.channel.id
    if storage_key not in chat_history or channel_id not in chat_history.get(storage_key, {}):
        await interaction.followup.send("ไม่พบประวัติการสนทนาค่ะ", ephemeral=True); return
    history_deque = chat_history[storage_key][channel_id]
    if len(history_deque) < 2 or history_deque[-1].get('role') != 'model' or history_deque[-2].get('role') != 'user':
        await interaction.followup.send("ไม่พบรูปแบบการสนทนาที่ถูกต้อง (User -> Model) เพื่อ reroll ค่ะ", ephemeral=True); return
    message_to_delete = None
    async for message in interaction.channel.history(limit=20):
        if message.author.id == bot.user.id: message_to_delete = message; break
    delete_success = False
    if message_to_delete:
        try: await message_to_delete.delete(); delete_success = True
        except discord.Forbidden: await interaction.followup.send("อายูมุไม่มีสิทธิ์ลบข้อความค่ะ!", ephemeral=True); return
        except discord.HTTPException as e: print(f"Failed to delete for reroll: {e}")
    history_deque.pop()
    msg_content = "รับทราบค่ะ กำลังคิดคำตอบใหม่..." if delete_success else "ลบข้อความเก่าไม่ได้ แต่กำลังคิดคำตอบใหม่ให้นะคะ..."
    await interaction.followup.send(msg_content, ephemeral=True)
    async with interaction.channel.typing():
        response_data = await generate_response(storage_key, channel_id)
        if response_data and response_data.get("reply"):
            final_message, thought_text, reply_text = "", response_data.get("thought", "").strip(), response_data.get("reply", "")
            if thought_text:
                formatted_thought = "\n".join([f"> {line}" for line in thought_text.split('\n')])
                final_message += f"{formatted_thought}\n"
            final_message += reply_text
            await interaction.channel.send(final_message)
        else:
            await interaction.channel.send("ขออภัยค่ะ อายูมุยังคิดคำตอบไม่ออกเลย >.<")

# <<< --- [เพิ่มใหม่] --- >>>
multimodal_group = discord.app_commands.Group(name="multimodal", description="ตั้งค่าการรับรู้ไฟล์ภาพและวิดีโอ")

@multimodal_group.command(name="toggleimage", description="เปิด/ปิดการประมวลผลรูปภาพที่แนบมากับข้อความ")
@discord.app_commands.choices(status=[discord.app_commands.Choice(name="On", value="on"), discord.app_commands.Choice(name="Off", value="off")])
async def toggleimage(interaction: discord.Interaction, status: discord.app_commands.Choice[str]):
    storage_key = get_storage_key(interaction)
    is_on = status.value == "on"
    server_settings.setdefault(storage_key, {})["process_images"] = is_on
    await interaction.response.send_message(f"รับทราบค่ะ! การประมวลผลรูปภาพถูก **{'เปิด' if is_on else 'ปิด'}** แล้วค่ะ", ephemeral=True)

@multimodal_group.command(name="togglevideo", description="เปิด/ปิดการประมวลผลวิดีโอ (ค่อนข้างนานนะคะ! ใช้อย่างระมัดระวัง)")
@discord.app_commands.choices(status=[discord.app_commands.Choice(name="On", value="on"), discord.app_commands.Choice(name="Off", value="off")])
async def togglevideo(interaction: discord.Interaction, status: discord.app_commands.Choice[str]):
    storage_key = get_storage_key(interaction)
    is_on = status.value == "on"
    server_settings.setdefault(storage_key, {})["process_videos"] = is_on
    await interaction.response.send_message(f"รับทราบค่ะ! การประมวลผลวิดีโอถูก **{'เปิด' if is_on else 'ปิด'}** แล้วค่ะ", ephemeral=True)

bot.tree.add_command(multimodal_group)
# <<< --- [จบส่วนที่เพิ่มใหม่] --- >>>

# (Personality, Debug, and Settings command groups are unchanged from your last version)
@bot.tree.command(name="editprompts", description="เปิดหน้าต่างเพื่อแก้ไข System Prompt และ Personality")
async def editprompts(interaction: discord.Interaction):
    storage_key = get_storage_key(interaction)
    
    user_settings = server_settings.get(storage_key, {})
    current_sys_prompt = user_settings.get("custom_system_prompt", system_instructions)
    current_personality = user_settings.get("custom_personality", personality_profile)

    # 1. สร้าง instance ของ Modal ขึ้นมาเปล่าๆ ก่อน
    modal = EditPromptsModal()

    # 2. **กำหนดค่า default ให้กับกล่องข้อความโดยตรง** << นี่คือหัวใจของการแก้ปัญหา
    modal.system_prompt_input.default = current_sys_prompt
    modal.personality_input.default = current_personality

    # 3. ส่ง Modal ที่ถูกตั้งค่าแล้วออกไป
    await interaction.response.send_modal(modal)
# <<< --- [เพิ่มใหม่] คำสั่งรีเซ็ต System Prompt --- >>>

@bot.tree.command(name="resetsysprompt", description="รีเซ็ต System Prompt กลับไปเป็นค่าเริ่มต้นจากไฟล์")
async def resetsysprompt(interaction: discord.Interaction):
    storage_key = get_storage_key(interaction)
    
    if storage_key in server_settings and "custom_system_prompt" in server_settings[storage_key]:
        del server_settings[storage_key]["custom_system_prompt"]
        await interaction.response.send_message(
            "✅ รีเซ็ต System Prompt เรียบร้อยแล้วค่ะ! อายูมุจะกลับไปใช้ค่าเริ่มต้นจากไฟล์ `system_prompt.txt` นะคะ",
            ephemeral=True
        )
    else:
        await interaction.response.send_message(
            "ℹ️ ตอนนี้อายูมุก็ใช้ System Prompt เริ่มต้นอยู่แล้วค่ะ ไม่มีอะไรให้รีเซ็ต",
            ephemeral=True
        )
@bot.tree.command(name="getcurrentpersonality", description="ดูบุคลิกปัจจุบันที่อายูมุใช้งานในเซิร์ฟเวอร์/DM นี้")
async def getcurrentpersonality(interaction: discord.Interaction):
    storage_key = get_storage_key(interaction)
    custom_personality = server_settings.get(storage_key, {}).get("custom_personality")
    title = "บุคลิกปัจจุบัน (ถูก Override):" if custom_personality else "บุคลิกปัจจุบัน (ค่าเริ่มต้นจากไฟล์):"
    current_personality = custom_personality or personality_profile
    await interaction.response.send_message(f"**{title}**\n```\n{current_personality}\n```", ephemeral=True)

@bot.tree.command(name="resetpersonality", description="รีเซ็ตบุคลิกของอายูมุกลับไปเป็นค่าเริ่มต้นจากไฟล์")
async def resetpersonality(interaction: discord.Interaction):
    storage_key = get_storage_key(interaction)
    if storage_key in server_settings and "custom_personality" in server_settings[storage_key]:
        del server_settings[storage_key]["custom_personality"]
        await interaction.response.send_message("✅ รีเซ็ตบุคลิกเรียบร้อยแล้วค่ะ!", ephemeral=True)
    else:
        await interaction.response.send_message("ℹ️ ตอนนี้อายูมุก็ใช้บุคลิกเริ่มต้นอยู่แล้วค่ะ", ephemeral=True)

@bot.tree.command(name="getlastresponse", description="ดึงข้อมูลดิบของคำตอบล่าสุดจาก Google AI (สำหรับดีบัก)")
async def getlastresponse(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    storage_key, channel_id = get_storage_key(interaction), interaction.channel.id
    raw_response = last_responses.get(storage_key, {}).get(channel_id)
    if not raw_response:
        await interaction.followup.send("ยังไม่มีข้อมูลคำตอบล่าสุดจาก AI ในแชทนี้เลยค่ะ", ephemeral=True); return
    dump_content = str(raw_response)
    dump_file = discord.File(fp=io.BytesIO(dump_content.encode('utf-8')), filename=f"last_response.txt")
    await interaction.followup.send("นี่คือข้อมูลดิบ (raw dump) ของคำตอบล่าสุดจาก Google AI ค่ะ:", file=dump_file, ephemeral=True)

@bot.tree.command(name="previewnextrequest", description="ดูข้อมูลดิบที่จะส่งไปให้ Google AI ในคำขอครั้งถัดไป (สำหรับดีบัก)")
async def previewnextrequest(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    storage_key, channel_id = get_storage_key(interaction), interaction.channel.id
    try:
        contents, model_kwargs = _prepare_request_payload(storage_key, channel_id)
        payload_to_dump = {"comment": "ข้อมูลที่จะถูกส่งไปยัง model.generate_content_async()", "contents": contents, "model_kwargs": model_kwargs}
        dump_content = json.dumps(payload_to_dump, indent=2, ensure_ascii=False)
        dump_file = discord.File(fp=io.BytesIO(dump_content.encode('utf-8')), filename=f"next_request_preview.json")
        await interaction.followup.send("นี่คือข้อมูลดิบที่จะถูกส่งไปให้ Google AI ในครั้งต่อไปค่ะ:", file=dump_file, ephemeral=True)
    except Exception as e:
        print(f"Error during previewnextrequest: {e}")
        await interaction.followup.send(f"เกิดข้อผิดพลาดขณะพรีวิวข้อมูล: {e}", ephemeral=True)

settings_group = discord.app_commands.Group(name="settings", description="ตั้งค่าการทำงานของบอท")
@settings_group.command(name="contextsize", description="ตั้งค่าขนาดประวัติการสนทนา (1-200000)")
@discord.app_commands.describe(size="จำนวนข้อความที่จะเก็บในประวัติ")
async def contextsize(interaction: discord.Interaction, size: discord.app_commands.Range[int, 1, 200000]):
    storage_key = get_storage_key(interaction)
    server_settings.setdefault(storage_key, {})["context_size"] = size
    await interaction.response.send_message(f"ตั้งค่า Context Size เป็น {size} แล้วค่ะ", ephemeral=True)

@settings_group.command(name="maxoutputtokens", description="ตั้งค่า Max Output Tokens (1-65536)")
@discord.app_commands.describe(tokens="จำนวน Token สูงสุดในการตอบกลับ")
async def maxoutputtokens(interaction: discord.Interaction, tokens: discord.app_commands.Range[int, 1, 65536]):
    storage_key = get_storage_key(interaction)
    server_settings.setdefault(storage_key, {})["max_output_tokens"] = tokens
    await interaction.response.send_message(f"ตั้งค่า Max Output Tokens เป็น {tokens} แล้วค่ะ", ephemeral=True)

@settings_group.command(name="messagelimit", description="ตั้งค่าจำนวนข้อความล่าสุดที่จะใช้ (1-100)")
@discord.app_commands.describe(limit="จำนวนข้อความล่าสุด")
async def messagelimit(interaction: discord.Interaction, limit: discord.app_commands.Range[int, 1, 100]):
    storage_key = get_storage_key(interaction)
    server_settings.setdefault(storage_key, {})["message_limit"] = limit
    await interaction.response.send_message(f"ตั้งค่า Message Limit เป็น {limit} แล้วค่ะ", ephemeral=True)

@settings_group.command(name="languagemodel", description="เลือกโมเดลภาษาที่จะใช้")
@discord.app_commands.choices(model=[
    discord.app_commands.Choice(name="Gemini 2.5 Flash-Lite Preview (เร็วฝุดๆ)", value="gemini-2.5-flash-lite-preview-06-17"),
    discord.app_commands.Choice(name="Gemini 2.5 Flash Preview (แนะนำ // เวอร์ชั่นทดลอง)", value="gemini-2.5-flash-preview-05-20"),
    discord.app_commands.Choice(name="Gemini 2.5 Pro Preview (ฉลาดมาก // เวอร์ชั่นทดลอง)", value="gemini-2.5-pro-preview-06-05"),
    discord.app_commands.Choice(name="Gemini 2.5 Flash (แนะนำ เวอร์ชั่นเสถียร)", value="gemini-2.5-flash"),
    discord.app_commands.Choice(name="Gemini 2.5 Pro (ฉลาดมาก เวอร์ชั่นเสถียร)", value="gemini-2.5-pro"),
])
async def languagemodel(interaction: discord.Interaction, model: discord.app_commands.Choice[str]):
    storage_key = get_storage_key(interaction)
    server_settings.setdefault(storage_key, {})["model"] = model.value
    await interaction.response.send_message(f"เปลี่ยนโมเดลเป็น `{model.name}` แล้วค่ะ!", ephemeral=True)

@settings_group.command(name="includethoughts", description="เปิด/ปิด การขอ 'กระบวนการคิด' จาก AI")
@discord.app_commands.choices(status=[discord.app_commands.Choice(name="On", value="on"), discord.app_commands.Choice(name="Off", value="off")])
async def includethoughts(interaction: discord.Interaction, status: discord.app_commands.Choice[str]):
    storage_key, is_on = get_storage_key(interaction), status.value == "on"
    server_settings.setdefault(storage_key, {})["include_thoughts"] = is_on
    await interaction.response.send_message(f"{'เปิด' if is_on else 'ปิด'}การขอ 'กระบวนการคิด' แล้วค่ะ", ephemeral=True)

@settings_group.command(name="thinkingbudget", description="ตั้งค่างบประมาณในการคิด (มีผลเมื่อเปิด includethoughts)")
@discord.app_commands.choices(level=[discord.app_commands.Choice(name="0 (ปิด)", value=0), discord.app_commands.Choice(name="1 (ต่ำ)", value=1024), discord.app_commands.Choice(name="2 (สูง)", value=-1)])
async def thinkingbudget(interaction: discord.Interaction, level: discord.app_commands.Choice[int]):
    storage_key = get_storage_key(interaction)
    server_settings.setdefault(storage_key, {})["thinking_budget"] = level.value
    await interaction.response.send_message(f"ตั้งค่างบประมาณการคิดเป็นระดับ {level.name} แล้วค่ะ", ephemeral=True)

bot.tree.add_command(settings_group)

@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: discord.app_commands.AppCommandError):
    if isinstance(error, discord.app_commands.CommandOnCooldown):
        await interaction.response.send_message(f"คำสั่งนี้กำลังคูลดาวน์ค่ะ รออีก {error.retry_after:.2f} วินาที", ephemeral=True)
    elif isinstance(error, discord.app_commands.errors.CommandInvokeError) and isinstance(error.original, discord.NotFound):
        print(f"Ignoring an expired interaction: {error.original.text}")
    else:
        print(f"Unhandled app command error: {error}")
        try:
            if not interaction.response.is_done(): await interaction.response.send_message("เกิดข้อผิดพลาดที่ไม่คาดคิดค่ะ", ephemeral=True)
            else: await interaction.followup.send("เกิดข้อผิดพลาดที่ไม่คาดคิดค่ะ", ephemeral=True)
        except discord.errors.InteractionResponded: print("Interaction already responded to during error handling.")
        except (discord.NotFound, discord.HTTPException) as e: print(f"Further error during error handling: {e}")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)