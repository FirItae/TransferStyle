import os 
import asyncio
import requests

from telebot import types
from telebot.async_telebot import AsyncTeleBot
from io import BytesIO
from PIL import Image

from wrapper import STTWrapper

token = os.environ.get('BOT_TOKEN')
bot = AsyncTeleBot(token)

@bot.message_handler(commands=['help', 'start'])
async def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn_start = types.KeyboardButton("Let's get started! 🖼️✨")
    markup.add(btn_start)

    await bot.reply_to(message, '''
            Hi there, I am StyleTransferBot! ✨🎨
I can help you transfer the style from one image to another.😊
            
1. First, I will ask you to send me an image of the content you want to stylize.
2. Once I receive the content image, I will ask you to send me an image with the style you want to apply.
3. After receiving the style image, I will work my magic and apply the style to the content image.
4. Finally, I will send you the stylized image for you to enjoy!

Please note that the style transfer process may take some time depending on the complexity of the images. But don't worry, I'll let you know when it's done!

If you're ready to proceed, just let me know
            ''' , reply_markup=markup)

# Handle the "Let's get started! 🖼️✨" message
@bot.message_handler(func=lambda message: message.text == "Let's get started! 🖼️✨")
async def start_style_transfer(message):
    await bot.reply_to(message, "Great! Please send me the image of the content you want to stylize.")
    # Set the state to expect the content image
    states[message.chat.id] = {"state": "expecting_content_image"}

# Handle the received images
@bot.message_handler(content_types=['photo'])
async def handle_images(message):
    chat_id = message.chat.id
    state = states.get(chat_id)
    if state["state"]  == "expecting_content_image":
        await process_content_image(message)
    elif state["state"]  == "expecting_style_image":
        await process_style_image(message)
    else:
        await bot.reply_to(message, "Sorry, I was not expecting an image at the moment.")

async def process_content_image(message):
    content_image = message.photo[-1] if message.photo else None
    if content_image:
        await bot.reply_to(message, "Awesome! Now, please send me the image with the style you want to apply.")
        # Set the state to expect the style image
        states[message.chat.id]["state"] = "expecting_style_image"
        states[message.chat.id]["content_image"] = content_image
    else:
        await bot.reply_to(message, "Please send me a valid image. Let's try again.")

async def process_style_image(message):
    style_image = message.photo[-1] if message.photo else None
    chat_id = message.chat.id
    state = states.get(chat_id)
    if style_image:
        await bot.reply_to(message, "Perfect! Now I'll work my magic and apply the style to the content image. Please wait a moment.")
        content_image = state.get("content_image")
        # Perform style transfer here using the content_image and style_image
        stylized_image =  await perform_style_transfer(content_image, style_image)
        if stylized_image:
            # Send the stylized image to the user
            await bot.send_photo(message.chat.id, photo=stylized_image)
            await bot.send_message(message.chat.id, "You can send another style and I will apply it too ")
            await bot.send_message(message.chat.id, "If you want to change the contextual image, click: Let's get started! 🖼️✨")
        else:
            await bot.reply_to(message, "Oops! Something went wrong during the style transfer process. Please try again.")
    else:
        await bot.reply_to(message, "Please send me a valid image. Let's try again.")


async def perform_style_transfer(content_image, style_image):
    content_image_bytes = requests.get(await bot.get_file_url(content_image.file_id))
    content_image_bytes = BytesIO(content_image_bytes.content)
    content_image_bytes.seek(0)
    content_image = Image.open(content_image_bytes)
    style_image_bytes = requests.get(await bot.get_file_url(style_image.file_id))
    style_image_bytes = BytesIO(style_image_bytes.content)
    style_image_bytes.seek(0)
    style_image = Image.open(style_image_bytes)
    out = model.predict(content_image,style_image)
    return out

# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(func=lambda message: True)
async def echo_message(message):
    await bot.reply_to(message, message.text)

if __name__ == '__main__':
    states = {} # Define states for the conversation
    model = STTWrapper()
    model.model.eval()
    asyncio.run(bot.polling())
