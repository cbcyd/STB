import numpy as np
from typing import List, Union
import asyncio
import aiofiles
import os
import time

import aiogram.utils
import aiogram.utils.markdown
import aiogram.types
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types import ParseMode, ContentType
from aiogram.utils import executor
from aiogram.dispatcher.handler import CancelHandler
from aiogram.dispatcher.middlewares import BaseMiddleware

from io import BytesIO
import concurrent.futures
import torch

from PAMA.main import mainPAMA

from magenta import load_image, transform, show

from NN.mainNN import loadNN, stylization
from NN.misc import USE_GPU

try:
    import nest_asyncio
    nest_asyncio.apply()
except:
    print('Running on script')
with open('token.txt') as f:
    bot = Bot(token=str(f.read()))
dp = Dispatcher(bot)
configDict = {}

pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
  
class AlbumMiddleware(BaseMiddleware):
    """This middleware is for capturing media groups."""

    album_data: dict = {}

    def __init__(self, latency: Union[int, float] = 0.01):
        self.latency = latency
        super().__init__()

    async def on_process_message(self, message: types.Message, data: dict):
        if not message.media_group_id:
            return

        try:
            self.album_data[message.media_group_id].append(message)
            raise CancelHandler()  # Tell aiogram to cancel handler for this group element
        except KeyError:
            self.album_data[message.media_group_id] = [message]
            await asyncio.sleep(self.latency)

            message.conf["is_last"] = True
            data["album"] = self.album_data[message.media_group_id]

    async def on_post_process_message(self, message: types.Message, result: dict, data: dict):
        """Clean up after handling our album."""
        if message.media_group_id and message.conf.get("is_last"):
            del self.album_data[message.media_group_id]

async def download_images(album):
    images_ids = []
    for obj in album:
        if obj.photo:
            file_id = obj.photo[-1].file_id
        else:
            file_id = obj[obj.content_type].file_id
        file = await bot.get_file(file_id)
        images_ids.append(file_id)
        await bot.download_file(file.file_path, 'photos/' + file_id + '.jpg')
    return images_ids

@dp.message_handler(commands='start')
async def start_command(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(KeyboardButton('Help'),KeyboardButton('Choose the model'))
    configDict[message.from_id] = 'Magenta'
    await message.reply("Hello there! I am a style transfer bot. Send me two images and I will transfer the style of the first image to the second image. You can choose from 3 models: <a href='https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'>Magenta</a>, <a href='https://github.com/luoxuan-cs/PAMA'>PAMA</a>, <a href='https://github.com/nkolkin13/NeuralNeighborStyleTransfer'>NeuralNeighbor</a>", reply_markup=keyboard, parse_mode="HTML")

@dp.message_handler(text='Help')
@dp.message_handler(commands='help')
async def help_command(message: types.Message):
    await message.reply(f"To use this bot, send me two images. I will transfer the style of the first image to the second image and send you back the stylized image. You can choose from 3 models: <a href='https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'>Magenta</a>, <a href='https://github.com/luoxuan-cs/PAMA'>PAMA</a>, <a href='https://github.com/nkolkin13/NeuralNeighborStyleTransfer'>NeuralNeighbor</a>. Current model is {configDict[message.from_id] if message.from_id in configDict else 'Magenta'}.", parse_mode="HTML")

@dp.message_handler(text='Choose the model')
async def config(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(KeyboardButton('Magenta'),KeyboardButton('PAMA'), KeyboardButton('NeuralNeighbor'))
    await message.reply('Choose your destiny', reply_markup=keyboard)

@dp.message_handler(text='Magenta')
@dp.message_handler(text='PAMA')
@dp.message_handler(text='NeuralNeighbor')
async def model(message: types.Message):
    configDict[message.from_id] = message.text
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(KeyboardButton('Help'),KeyboardButton('Choose the model'))
    await message.reply(f'You have chosen {message.text} model',reply_markup=keyboard)

async def delete(images_ids):
    os.remove('photos/' + images_ids[1] + '.jpg')
    os.remove('photos/' + images_ids[0] + '.jpg')
    os.remove('photos/' + images_ids[0] + images_ids[1] + '.jpg')

async def runMagenta(message, album):
    msg = await message.reply('Loading album')
    images_ids = await asyncio.create_task(download_images(album))

    await msg.edit_text('Loading images')
    content_image, style_image = await asyncio.create_task(load_image('photos/' + images_ids[1] + '.jpg', 'photos/' + images_ids[0] + '.jpg'))

    await msg.edit_text('Stylizing image')
    startTime = time.time()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        loop = asyncio.get_running_loop()
        stylized_image = await loop.run_in_executor(pool, transform, content_image, style_image)
    endTime = time.time()

    await msg.edit_text('Transforming image')
    await asyncio.create_task(show(stylized_image, images_ids))
    
    await msg.edit_text('Sending image')
    async with aiofiles.open('photos/' + images_ids[0] + images_ids[1] + '.jpg', mode='rb') as f:
        await bot.send_photo(message.chat.id, photo=f, reply_to_message_id=message.message_id, caption=f'Stylization took {endTime - startTime} seconds')
    
    await msg.delete()
    await asyncio.create_task(delete(images_ids))
    
    try: torch.cuda.empty_cache()
    except: ''

async def runPAMA(message, album):
    msg = await message.reply('Loading album')
    images_ids = await asyncio.create_task(download_images(album))

    await msg.edit_text('Stylizing image')
    startTime = time.time()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(pool, mainPAMA, 'photos/' + images_ids[1] + '.jpg', 'photos/' + images_ids[0] + '.jpg')
    endTime = time.time()

    await msg.edit_text('Sending image')
    async with aiofiles.open('photos/' + images_ids[0] + images_ids[1] + '.jpg', mode='rb') as f:
        await bot.send_photo(message.chat.id, photo=f, reply_to_message_id=message.message_id, caption=f'Stylization took {endTime - startTime} seconds')
    
    await msg.delete()
    await asyncio.create_task(delete(images_ids))

    try: torch.cuda.empty_cache()
    except: ''

async def runNN(message, album):
    if USE_GPU: torch.cuda.empty_cache()

    msg = await message.reply('Loading album')
    images_ids = await asyncio.create_task(download_images(album))
    content_im_orig, style_im_orig, output_path = await asyncio.create_task(loadNN(images_ids))
    
    await msg.edit_text('Stylizing image')
    loop = asyncio.get_running_loop()
    timeNN = await loop.run_in_executor(pool, stylization, content_im_orig, style_im_orig, output_path)

    await msg.edit_text('Sending image')
    async with aiofiles.open('photos/' + images_ids[0] + images_ids[1] + '.jpg', mode='rb') as f:
        await bot.send_photo(message.chat.id, photo=f, reply_to_message_id=message.message_id, caption=f'Stylization took {timeNN} seconds')

    await msg.delete()
    await asyncio.create_task(delete(images_ids))

models = {'Magenta': runMagenta, 'PAMA': runPAMA, 'NeuralNeighbor': runNN}

@dp.message_handler(is_media_group=True, content_types=types.ContentType.ANY)
async def process_image(message: types.Message, album: List[types.Message]):
    if message.from_id not in configDict:
        configDict[message.from_id] = 'Magenta'

    await models[configDict[message.from_id]](message, album)

if __name__ == '__main__':
    dp.middleware.setup(AlbumMiddleware())
    executor.start_polling(dp, skip_updates=True)