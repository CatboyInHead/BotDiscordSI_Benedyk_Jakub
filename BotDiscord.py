import discord
from discord.ext import commands
import io
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from unittest.mock import patch

token = "MTUxNjAwMTg5MDMzNTE5OTM0NA.GnjDkm.Gb7ekd4rL6RvOJnnXG8cDbFlW4SzeqyQIcxOAY"

print("Ładowanie modelu Florence-2... Może to chwilę potrwać.")
model_id = "microsoft/Florence-2-base"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

with patch("transformers.dynamic_module_utils.check_imports"):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch_dtype).to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    print(f'Bot zalogowany jako: {bot.user.name}')


@bot.command()
async def rozpoznaj(ctx):
    if not ctx.message.attachments:
        await ctx.send("Musisz dodać obrazek jako załącznik do tej komendy!")
        return

    attachment = ctx.message.attachments[0]

    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
        status_msg = await ctx.send("Otrzymałem obraz. Przyglądam się szczegółom...")

        try:
            image_bytes = await attachment.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            prompt = "<MORE_DETAILED_CAPTION>"

            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            parsed_answer = processor.post_process_generation(generated_text, task=prompt,
                                                              image_size=(image.width, image.height))
            opis_en = parsed_answer[prompt]

            wynik = f"**Oto szczegółowy opis tego, co widzę:**\n{opis_en}"

            await status_msg.edit(content=wynik)

        except Exception as e:
            await status_msg.edit(content=f"Wystąpił błąd podczas analizy: {e}")

    else:
        await ctx.send("Niestety, to nie jest poprawny format obrazu.")

bot.run(token)
