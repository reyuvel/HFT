import os
import json
import requests
import io
import re
from pathlib import Path
from telegram import Update
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext,CallbackQueryHandler
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from pydub import AudioSegment
from dotenv import load_dotenv
from openai import OpenAI
import urllib.parse
import time

# Load environment variables
load_dotenv()
voic=False
openai_api_key = os.getenv("OPENAI_API_KEY")
telegram_api_key = os.getenv("TOKEN")
google_api_key = os.getenv("GOOGLE_API_KEY")
device_selectedd = True

if not openai_api_key:
    raise ValueError("OpenAI API key not found.")
if not telegram_api_key:
    raise ValueError("Telegram API key not found.")
if not google_api_key:
    raise ValueError("Google API key not found.")
previous_device_name = ""


def send_audio(update: Update, context: CallbackContext) -> None:
    AUDIO_FILE_PATH = '/home/dias/Documents/WhisperAI/speech.mp3'
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"Audio file does not exist at {AUDIO_FILE_PATH}")
    try:
        context.bot.send_audio(chat_id=update.effective_chat.id, audio=open(AUDIO_FILE_PATH, 'rb'))
    except Exception as e:
        print(f"Error sending audio: {e}")


def generate_prompt(description, product_keywords):
    prompt = "User Description: " + description
    if product_keywords:
        prompt += " " + ", ".join(product_keywords)
    return prompt

def train_model(prompt, context, update):    
    chat_model = ChatOpenAI(
        temperature=0,  # Adjust temperature as needed
        model='gpt-4',  # Specify the model (e.g., 'gpt-4')
        openai_api_key=openai_api_key,
        max_tokens=350
    )
    output = chat_model([
        HumanMessage(content=context),  # Provide context
        HumanMessage(content=prompt)  # Provide prompt
    ])

    response = output.content
    print(response)
    update.message.reply_text(response)   #used to send text message
    user = update.message.from_user
    global device_selectedd
    if (device_selectedd): 
            keyboard = [
        [InlineKeyboardButton("Select a Device", callback_data='select_device')]
        
            ]
            device_selectedd = False

    else:
            keyboard = [
       [InlineKeyboardButton("Confirm Your Device", callback_data='confirm_device')]
            ]
    

    reply_markup = InlineKeyboardMarkup(keyboard)

    response2 = update.message.reply_text(f'Hi {user.first_name}', reply_markup=reply_markup)

    return response


def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Hi! I am your chatbot. Send me a message.')

def echo(update: Update, context: CallbackContext) -> None:
    global previous_device_name
    
    # Get user message
    user_message = update.message.text

    # Check if a device name is already selected
    if context.user_data.get("device_selected", False):
        # If a device is already selected, treat the message as a description
        response = process_message(user_message)
        update.message.reply_text(response)
    else:
        # If no device is selected, prompt the user to select a device first
        update.message.reply_text("Please use the /selectdevice command first to select a device.")


def send_device_selection_message(update, context):
    # Create inline keyboard buttons for device selection
    keyboard = [
        [InlineKeyboardButton("Device 1", callback_data='device_1')],
        [InlineKeyboardButton("Device 2", callback_data='device_2')],
        [InlineKeyboardButton("Device 3", callback_data='device_3')],
        # Add more buttons as needed
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Send message with inline keyboard
    update.message.reply_text('Please select a device:', reply_markup=reply_markup)


def button_click(update, context):
    query = update.callback_query
    device = query.data  # Get the device selected by the user from the callback data
    # Perform actions based on the selected device (e.g., fetch device details)
    fetch_device_details(device, update, context)
    # You can also edit the message to provide feedback to the user
    query.edit_message_text(f"You selected: {device}")


def process_message(message, update: Update, context: CallbackContext):
    # Process the message using your chatbot logic
    # For simplicity, let's assume we directly use the chatbot model here
    # You might want to update this part based on your chatbot implementation
    description = message
    product_keywords = extract_product_keywords(description)
    prompt = generate_prompt(description, product_keywords)
    context_message = "Context: Orca is an AI assistant that provides recommendations about devices based on user requirements."
    response = train_model(prompt,context_message)
    print(response)

    update.message.reply_text(response)
    print(response)
    print("----resp---", response)
    return response



def extract_product_keywords(description):
    # Read keywords from keywords.txt
    with open("keywords.txt", "r") as file:
        relevant_keywords = [line.strip() for line in file]

    # Check if any product keywords are present in the description
    found_keywords = [keyword for keyword in relevant_keywords if keyword in description]
    print(found_keywords)
    return found_keywords

def select_device(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()
    query.message.reply_text("Please record/type the name of the device you want to buy.")

def handle_device_name(update: Update, context: CallbackContext) -> None:
    global previous_device_name
    global device_selectedd
   
    # Check if a device name has already been selected
    if context.user_data.get("device_selected", True):
        # If a device is already selected, treat the message as a description
        description = update.message.text
        previous_device_name=description
        print(description)
        product_keywords = extract_product_keywords(description)
        prompt = generate_prompt(description, product_keywords)
        context = "Context: Orca is an AI assistant that provides recommendations about devices based on user requirements."
        train_model(prompt, context, update)
        # update.message.reply_text(response)
    else:
        # If no device is selected, treat the message as a device selection
        device_name = update.message.text.strip()
        previous_device_name = device_name
        context.user_data["device_selected"] = True
        update.message.reply_text(f"Device name '{previous_device_name}' has been selected.")



def confirm_device(update: Update, context: CallbackContext) -> None:
    global previous_device_name
    query = update.callback_query
    query.answer()
    if previous_device_name:
        fetch_device_details(previous_device_name, update, context)  # Pass device_name and send_function
    else:
        update.message.reply_text("No device name selected. Please use the /selectdevice command first.")


def fetch_device_details(device_name,update,context,delay=1):
    if  not device_name:
        print("No device name selected. Please use the /selectdevice command first.")
    # Format the device name to replace spaces with '+' and handle special characters
    #formatted_device_name = device_name.replace(' ', '')
    formatted_device_name = urllib.parse.quote(device_name.strip(), safe='+')
    
    print(formatted_device_name)
    # Send request to Google SERP API's Shopping API
    url = f"https://serpapi.com/search?engine=google_shopping&q={formatted_device_name}&api_key={google_api_key}&gl=in&img=1"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        details = data.get('shopping_results')
        if details:
            trusted_platforms = ["Amazon","Flipkart"] 
            trusted_results = [item for item in details if item.get('source') in trusted_platforms]
            if trusted_results:
                sorted_trusted_results = sorted(trusted_results, key=lambda x: x.get('price', float('inf')))
                result = sorted_trusted_results[0]
                platform = result.get('source')
                price = result.get('price')
                link = result.get('link')
                message = f"Platform: {platform}\nPrice: {price}\nURL: {link}\n\n"
                context.bot.send_message(chat_id=update.effective_chat.id, text=message)
                time.sleep(delay)
            # return "\n".join(device_info)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching device details: {e}")

    return None

def send_function(message):
    print("Sending:", message)


client = OpenAI()

def convert_bytesio_to_mp3(audio_io: io.BytesIO, mp3_filename: str):
    audio_io.seek(0)  # Reset the pointer of the BytesIO object to the beginning
    audio_data = AudioSegment.from_file(audio_io, format="ogg")  # Specify the format if known
    audio_data.export(mp3_filename, format='mp3')  # Export as MP3

def process_voice(update: Update, context: CallbackContext) -> None:
    
    file = context.bot.getFile(update.message.voice.file_id)


    audio_data = io.BytesIO()
    file.download(out=audio_data)

    try:
        # Convert the downloaded voice message to an MP3 file
        mp3_filename = "voice_message.mp3"
        convert_bytesio_to_mp3(audio_data, mp3_filename)

        # Transcribe the audio file using OpenAI's Whisper model
        with open(mp3_filename, "rb") as audio_file:
            transcript = client.audio.translations.create(
                model="whisper-1",
                file=audio_file
            
            )
        # print("trans",transcript.language)
        result = transcript.text # Assuming that the response has a 'text' field
        print(result)
        # text = extract_content(result)

        print("vannn",result)
        
        # Instead of replying with the transcription, handle it as if it was a text message
        update.message.text = result # Set the transcribed text as if it were a regular text message
        handle_device_name(update, context) # Use the transcribed text as if it were user-typed text
        
    except Exception as e:
        update.message.reply_text('An error occurred while processing the audio.')
        print(f"Error: {e}")

def main() -> None:
    global voic
    # Initialize the Telegram Bot
    updater = Updater(telegram_api_key)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher
    # process_voice_try=process_voice(updater)
    
    # Register handlers
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(select_device,pattern='select_device'))
    dp.add_handler(CallbackQueryHandler(confirm_device,pattern='confirm_device'))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command & ~Filters.update.edited_message, handle_device_name))
    dp.add_handler(CommandHandler("device", send_device_selection_message))
    dp.add_handler(CallbackQueryHandler(button_click))
    dp.add_handler(MessageHandler(Filters.voice, process_voice))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()