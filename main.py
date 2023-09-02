from typing import Final
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler, CallbackContext

from model import main


TOKEN: Final = "6391484366:AAEvPRoDj00YHUzYuutaFcut2t7qtP5MhXw"
BOT_USERNAME: Final = "@VeryDeepStyleBot"


STYLE, SOURCE = range(2)
count = 0

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello, Sir! Send me a style photo')

    return STYLE

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('You can send two photos and I\'ll transfer the style from the first photo to the second one.')

async def style_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download_to_drive(f"style_photo{count}.jpg")
    print("Photo of %s: %s", user.first_name, f"style_photo{count}.jpg")
    await update.message.reply_text('Now send me a source photo')

    return SOURCE

async def source_photo(update: Update, context: CallbackContext):
    global count
    user = update.message.from_user
    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download_to_drive(f"source_photo{count}.jpg")
    print("Photo of %s: %s", user.first_name, f"source_photo{count}.jpg")
    await update.message.reply_text('Now I will generate a new one!')

    main(f"source_photo{count}.jpg", f"style_photo{count}.jpg", count)

    await context.bot.send_document(chat_id=update.message['chat']['id'], document=open(f"result{count}t.png", 'rb'), filename=f"result{count}t.png")
    count += 1

    return ConversationHandler.END


def handle_response(text: str) -> str:
    processed: str = text.lower()

    if 'hello' in processed:
        return 'hey'
    
    if 'how are you?' in processed:
        return 'I am perfect!'
    
    return 'Send me a photo finally!'


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type: str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}: "{text}"')

    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)
        else:
            return
    else:
        response: str = handle_response(text)

    print('Bot: ', response)
    await update.message.reply_text(response)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    print("User %s canceled the conversation.", user.first_name)
    await update.message.reply_text(
        "Good luck!", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END


if __name__ == '__main__':
    app = Application.builder().token(TOKEN).build()


    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            STYLE: [MessageHandler(filters.PHOTO, style_photo)],
            SOURCE: [MessageHandler(filters.PHOTO, source_photo)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )


    # Commands
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler('help', help_command))


    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Log all errors
    app.add_error_handler(error)

    print('Polling...')
    # Run the bot
    app.run_polling(poll_interval=5)

