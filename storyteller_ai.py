import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import openai
import random

#Narrator role
#input: the 6 card in his hands

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
openai.api_key = 'KEY'

def embedding_extraction(card):
    cards_path = "cards/" + card
    img = Image.open(cards_path)
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
    image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
    return image_embedding

# Function to get the closest description using CLIP
def get_best_matching_description(image_embedding, descriptions):
    inputs = processor(text=descriptions, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs)
    text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)
    similarity = torch.mm(image_embedding, text_embeddings.T)
    best_match_idx = torch.argmax(similarity).item()
    best_description = descriptions[best_match_idx]
    return best_description

# Function to generate clues with GPT-3.5
def GPT_generation(description):
    prompt = f"I have an image described as: '{description}'. Imagine you are the storyteller in a Dixit game. Provide five different very short clues, no more than four words each."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a creative storyteller giving cryptic and evocative clues, based on a visual description."},
                  {"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.9,
        top_p=1.0
    )
    clues = response['choices'][0]['message']['content'].strip()
    return clues

# Main function to generate a clue for a specific card
def generate_hint(cards):
    descriptions = [
        "A young girl in a white dress floating in the air, holding onto giant dandelions like balloons, surrounded by a dreamy, cloudy sky.",
        "A wooden house built between two cliffs, lit by warm lights, with a person descending in a small suspended cable car. Below, a massive monster wolf with glowing red eyes roars towards a floating lantern, under the moonlit sky.",
        "A tiny cozy house with a lit fireplace, nestled in a forest with a giant tree. The tree is carved with mysterious faces, and the scene is calm and peaceful, bathed in warm yellow light.",
        "A person sitting on a cushion, reading a book in front of a fireplace, seemingly floating high up on a wall covered with old paintings. A second figure is reading a book far below, as sunlight streams through a high window.",
        "A large bouquet of orange roses in a vase, shaped like a woman's torso, stands by a window overlooking a snowy street with a bridge and houses. A small candle burns on the windowsill, creating a warm, cozy contrast with the cold winter scene outside.",
        "A young traveler with a large ball strapped to their back, standing in a mysterious maze-like landscape. They are observing a glowing plant with delicate, bell-shaped flowers, while an eerie greenish light illuminates the scene.",
        "An open book surrounded by red flowers, with delicate white vines growing out of its pages. A white bird takes flight among the vines, creating a sense of magic and wonder against a deep red background.",
        "Two figures standing on a bench, looking up at giant whales swimming between tall, bare trees in a misty forest. The entire scene feels otherworldly and surreal, as if underwater.",
        "A playful robot made from a tomato can and other household items, holding a spoon and wearing a small patterned box as its head. The background is a warm and colorful interior space.",
        "A man walking on a tightrope between two cliffs, carrying a balancing pole, while a crane hovers above, holding a large stone block to complete the broken path. The landscape below features a river and green fields under a bright blue sky.",
        "A small boy sleeping in a bed that floats on water during a rainstorm, while a man in a boat rows above a whale. Giant fishes swim nearby, giving the sense of a magical, dream-like setting.",
        "A tall cliffside with a collection of small wooden houses stacked upon one another, leading up to a grassy plateau with a single tree. On the underwater side, a large whale floats, creating an ethereal contrast between land and water.",
        "A ball of yarn with the appearance of a watermelon, with a piece of yarn trailing off. Behind it there is the shadow of a cat paw on the background. The background is dark, adding a mysterious, almost eerie feeling to the scene.",
        "A child in a quiet street, while a gust of wind carries a flurry of orange leaves out of a dark alleyway, creating the appearance of a magical, animated creature emerging from the shadows.",
        "A narrow, cobblestone street in a European city at dusk, with warm lights glowing from windows and lanterns. The street leads to a grand domed building in the distance, creating a cozy yet mysterious atmosphere.",
        "A grand piano with a whimsical twist: its interior reveals a miniature landscape with a river winding through fields, trees, and iconic landmarks. The room also features a framed painting of the Mona Lisa, adding to the surreal quality of the scene.",
        "An old king with a long beard sitting on an ornate golden throne. His beard flows down like a river, with a small goldfish swimming along it, adding a magical and almost fairy-tale-like element to the image.",
        "Two children peeking at each other on a spiraling staircase. One child in a red dress stands at the top, while the other is lower down, creating a sense of playfulness and curiosity.",
        "Two foxes sitting in a cozy underground burrow, sharing a meal by candlelight. Outside, the winter landscape is covered in snow, with a dark starry sky and tall pine trees, creating a warm contrast between the underground and the cold world above.",
        "A small sailboat floating on a sea of blue gears and cogs, with a tiny sun and cloud in the sky above. The mechanical sea gives a surreal and imaginative twist to a traditional sailing scene.",
        "A young girl sitting atop a high stack of mattresses and blankets, holding a glowing light. Small animals are hiding between the layers, while her shadow creates an imaginative silhouette on the wall.",
        "A Santa Claus figure with a long blue beard, standing atop a chimney under a snowy sky, holding a small wrapped present in one hand. The scene conveys a whimsical holiday atmosphere.",
        "A cat sitting on a rooftop, looking up at a giant moon that appears to be a ball of yarn. The buildings are illuminated by moonlight. The night sky is filled with stars, giving a magical and playful twist to the scene.",
        "Two children walking hand in hand, casting a shadow that resembles a fierce wolf. The shadow contrasts their innocent appearance, creating an eerie and suspenseful mood.",
        "A young girl in a white dress standing on a staircase, gazing lovingly at a smiling moon with tendrils extending like rays. The night sky is full of clouds, giving a dreamlike and serene atmosphere.",
        "A birdcage that looks like a head on a torso and has a human eye in the center, surrounded by colorful birds. One bird holds a golden key in its beak, adding a surreal and mysterious element to the scene.",
        "A little man wearing a nightgown coming out from a crescent moon, using strings to control a small hanging lantern. The entire scene has a whimsical and dreamlike quality.",
        "A tree with a pencil-like body, sharpened in the middle, standing among stumps of other cut-down trees. A man with an axe looks at the tree, contemplating his action.",
        "A man and a woman on a small floating island. The man is scattering letters in the air, while the woman is watering plants that have pages of text growing from them.",
        "A child standing on the edge of a cliff, holding strings connected to the stars and a crescent moon as if they were balloons. The scene has a sense of wonder, magic, and childlike curiosity.",
        "A little girl standing at the base of a large pile of presents, holding a small wrapped gift. At the top of the pile sits a boy, surrounded by the various boxes, creating a playful scene of giving and receiving.",
        "A tree with its branches transforming into the form of a bird, its beak is a pencil and is drawing a red heart on a piece of paper. Another small bird sits in the upper branches.",
        "A girl wearing a red coat is growing out of a flowerpot, holding an umbrella from which rain falls.",
        "A man dressed in black with a top hat walks a tightrope while balancing two weights: a large human heart on one side and a brain on the other. The background is the night with stars. The image symbolizes the balance between emotions and reason.",
        "A small, cozy house in a dark, snowy forest, with light glowing warmly from its windows. A figure in red stands on a narrow bridge leading to the house, surrounded by towering blue trees, creating a mysterious yet inviting scene.",
        "A fish swimming inside a birdcage filled with water, sitting on top of a wooden dresser. The unusual juxtaposition of a fish in a cage adds a surreal and thought-provoking element to the scene.",
        "A girl playing the violin while sitting on an extremely tall chair, surrounded by floating chairs and scattered music sheets. There is also a door in the background. The scene has a dreamy, whimsical atmosphere, as if the objects are weightless and suspended in time.",
        "An elaborate mechanical observatory with an intricate array of devices, gears, and a large telescope. The space is lit warmly, filled with a sense of wonder and discovery, evoking the atmosphere of an inventor’s workshop.",
        "A wooden pier stretching out into the moonlit ocean, surrounded by glowing orbs of light. The path leads toward the moon, which casts a shimmering reflection on the water, creating a magical and tranquil scene.",
        "A man in a white robe with a crown, standing on a large chessboard platform atop a tower. The platform seems to float in a surreal red sky, giving the impression of a king in an otherworldly setting.",
        "A tree stump with a bright red heart-shaped cut, surrounded by lush green foliage. A red flower blooms next to the stump.",
        "A young girl sitting on a grassy hilltop, holding a long stick with which she reaches up to the stars. In the background, there is a city skyline of tall buildings.",
        "A mysterious figure in a dark, crescent-shaped boat, fishing in the sky using a large net to capture stars and the moon. The entire scene is bathed in a blue monochrome light, creating a magical and surreal atmosphere.",
        "A whimsical scene of a majestic elephant carrying a tall, ornate building on its back. There's small and strange figure atop the elephant. A cat sits on the moon. The sky is filled with clouds, giving the impression of a fantastical journey through a dream.",
        "A young girl with long hair sits at the edge of a floating hill, overlooking an ethereal landscape of cone-shaped houses. The scene is filled with a sense of enchantment and the beauty of a magical world.",
        "A house on tall stilts, with a giant child's face looking out of a window. The child's hand appears to be reaching through the side of the building, creating a surreal and slightly eerie visual, as if the house were a living entity.",
        "An old, moss-covered robot head sitting quietly in a dark forest. Blue butterflies surround it, adding a sense of melancholy and beauty, as if nature is reclaiming the abandoned machine.",
        "A single person walks along a narrow path, surrounded by towering, twisted trees that form a canopy overhead. The atmosphere is eerie yet mesmerizing, with rays of light peeking through, casting a glow on the traveler. The size difference between the trees and the person conveys a sense of wonder and insignificance.",
        "A young girl, dressed in a pink leotard, emerges gracefully through the cracks of ice as if performing a ballet routine. The light blue background and subtle lighting convey an aura of innocence, hope, and fragility.",
        "Figures are suspended in mid-air, each tethered to the ground by long, thin strings. The scene features quaint red-roofed houses beneath a light blue sky, creating an impression of humans soaring like balloons.",
        "A small, wooden cottage stands in darkness, enveloped by mist. Three large ghostly figures, with bright yellow eyes, loom in the background, casting an ominous presence over the house. The imagery evokes mystery, fear, and suspense.",
        "A young boy, wearing a graduation cap, gazes thoughtfully at a paper boat. Next to him, a small stuffed mouse is also dressed in a similar cap. Mathematical formulas fill the background, suggesting themes of childhood curiosity, learning, and imagination.",
        "A young boy nervously holds a bouquet of colorful flowers behind his back, looking toward a girl dressed in an elegant white gown. She appears unaware, standing gracefully with her back turned. The pink background evokes a sense of innocence, admiration, and budding romance.",
        "A parchment scroll filled with handwritten text curls at the ends, with a small flower blooming from the middle.",
        "A library setting with bookshelves towering towards a round window. An open book is floating in mid-air, seemingly enchanted. The soft light shining through the window gives the scene a sense of wonder and mystery.",
        "A solitary figure sits at a small table, seemingly having tea, under the glowing lights of traffic signals. The surrounding trees, illuminated in muted blues, create a surreal and dream-like environment, blending the urban and natural elements into a reflective, solitary atmosphere.",
        "An anthropomorphic fox in a blue tunic is seen carrying a small wooden house on its back. The house is whimsically adorned with flower decorations and warm lights. The lush greenery around gives the image a sense of determination and fantasy.",
        "A surreal image of a man's head morphing into a castle tower. The tower has small windows, and a tall sword with a red drop of blood that sits at the top. The dark, cloudy background.",
        "A whimsical scene featuring a character sitting atop a signpost in the rain. The signpost is labeled with directions like Nowhere, Here, There, Everywhere, and Somewhere. The character holds an umbrella.",
        "A young woman with long hair sits in a small patch of greenery surrounded by a sprawling cityscape of tall gray buildings. The juxtaposition of the natural and urban environments creates a sense of isolation yet hope, as if the greenery is a small sanctuary amidst the concrete jungle.",
        "An enchanted forest scene where massive trees have hollow interiors, with windows and doors carved into their trunks, suggesting they serve as homes. The soft golden light emanating from within gives a cozy and mystical ambiance, creating a magical atmosphere that hints at hidden stories and forest dwellers.",
        "A man with large shears is cutting his way through a towering green labyrinth, creating a direct path. Ahead of him, a woman is waiting at the end of the newly cut path.",
        "A girl in a white dress stands on a wooden bench that is partially submerged in water. A bright red umbrella lies open beside her, contrasting the tranquil color palette of the sky and water. The image evokes feelings of solitude, contemplation, and a sense of being adrift in a serene environment.",
        "A mysterious figure cloaked in black plays a violin in front of a sleeping blue dragon resting on a pile of gold coins. The scene evokes a blend of fantasy and magic, suggesting that music has lulled the dragon to sleep. The dungeon-like background and the golden treasure add to the atmosphere of myth and legend.",
        "A young woman sits in a small wooden boat, rowing through an underwater cityscape, with skyscrapers visible beneath the surface. She seems lost in thought, surrounded by white flowers floating on the water.",
        "An owl judge presides over a courtroom, raising a gavel while a rabbit, a wolf, and a pig stand in front of it with placards displaying their identification numbers. The scene resembles a court trial.",
        "An old, broken-down fishing boat sits abandoned on a beach at sunset. Above the boat, fluffy clouds take the shape of a majestic sailing ship with full sails, glowing warmly in the evening light. The scene portrays the contrast between reality and imagination, suggesting the lingering spirit of adventure despite the current state of the boat.",
        "A young boy stands on a grassy field, holding a garden hose that sprays water into the sky. The water magically turns into a large, fully-rigged sailing ship, as if the boy is powering the vessel with his hose.",
        "A woman sits atop a massive bird with blue feathers, holding a glowing orb. The bird's wings are covered with architectural structures resembling buildings and castles, giving the impression of a city in flight. The moon glows in the background and it looks like a clock.",
        "A bunny wearing clothes is sitting on a pink ladder, reading a book while surrounded by hanging laundry.",
        "A man with a cage-like top hat stands in profile. Inside the hat, two white doves sit calmly. Another face appears subtly in the background, blended into the sky.",
        "Two children are riding in a small red car, joyfully raising their hands as they soar over the peak of a hill. The road and surrounding landscape are simplified and bathed in warm light, giving the scene a sense of carefree adventure, freedom, and childhood exuberance.",
        "The façade of a building with several windows is depicted during snowfall. In one window, a person plays a guitar, while another person leans out of a different window, enjoying the scene. The muted blue tones and falling snowflakes give the image a serene, cozy, and reflective atmosphere, highlighting moments of human connection in an urban setting.",
        "A child sits on a swing suspended in a room filled with magical elements. Two large, fish-like creatures with ornate patterns emerge from the walls, while strings with stars and crescent moons hang around. The child appears calm, and the whole room has a dream-like quality, evoking a sense of wonder, fantasy, and introspection.",
        "A cityscape floats on top of a large black balloon, tethered by strings to a smaller Earth below. The bright sky and puffy clouds contrast with the darker tones of the city, creating an intriguing juxtaposition.",
        "A child bundled in winter clothes is playing chess with a giant snowman. The snowman, with its exaggerated carrot nose and long stick arms, bends down to make its move. The scene is set in a snowy landscape.",
        "A young girl stands in awe in the middle of an enormous library filled with towering bookshelves, stretching high above her. She holds a book while a black cat sits beside her on a pile of books. The warm, golden lighting enhances the sense of wonder and curiosity, as if she is surrounded by a world of endless knowledge waiting to be discovered.",
        "A child and a dog are lying in the snow, joyfully making snow angels. The child, dressed warmly, is smiling while their briefcase lies beside them. The dog mimics the child, creating its own snow angel.",
        "An hourglass containing two distinct halves: the top half is filled with water and a fish, while the bottom half holds a bird standing on dry sand. A single droplet falls from the top, connecting the two environments.",
        "A collection of abstract, colorful blocks stacked in a whimsical arrangement. Some of the blocks are open frameworks, while others are solid with geometric patterns and spirals. The playful composition and vibrant colors give the impression of a fantastical building, reminiscent of a toy structure or a puzzle waiting to be solved.",
        "A person stands on a hill, shining a flashlight toward the moon. The illuminated area reveals that part of the moon is actually made of red bricks, as if exposing a hidden reality beneath its surface.",
        "A red dragon, with its body halfway through an arched window, appears to be entering a stone-walled building, maybe a tower. Its bright, textured scales stand out against the muted tones of the wall.",
        "A silhouetted figure in a top hat rides an old-fashioned penny-farthing bicycle across a tightrope. The enormous full moon behind them makes the figure appear as if cycling on the moon itself. Below, a quaint town is silhouetted against the night sky.",
        "A mysterious, dark-haired woman plays an instrument that looks like a cello. Her long hair stretches out like musical notes, with children appearing to be carried along by the strands, as if they are flowing with the music. The underwater setting and the dreamy expressions on the children’s faces give the scene a magical, almost eerie feeling.",
        "A startled hedgehog holding a bow, looking at a chaotic web of scratched lines on a red background.",
        "In a dark forest with tall, thin trees, a small horse-drawn carriage is parked. The warm glow of a lantern illuminates the area, casting light on the carriage and creating a mysterious, enchanting atmosphere. The forest seems dense and almost surreal.",
        "An elderly man with sunglasses and a cane sits on a park bench, offering his hand to a large stork standing in front of him. The sky is filled with birds in flight.",
        "A large black creature with glowing eyes peers through an open door, casting a long shadow into the room. The creature looks curious yet somewhat intimidating, with the perspective emphasizing its size compared to the doorway.",
        "A young child dressed in a yellow raincoat stands confidently at the wheel of a ship, navigating stormy seas. Behind them, a vibrant rainbow arches across the sky, contrasting with the dark waves.",
        "A young girl floats up into the night sky, holding a red umbrella. She is dressed in a white dress and wears bright red shoes. The background features a glowing full moon and quaint rooftops below, as glowing stars gently fall around her like raindrops.",
        "A young girl stands in a dimly lit room, looking out of a large window. Outside, two gigantic fish with curious expressions seem to be floating in mid-air, staring into the room. The soft, underwater-like lighting and the girl's calm stance give the scene an eerie yet peaceful atmosphere.",
        "A child in a blue winter coat gleefully runs with a butterfly net, chasing a cluster of floating Santa Claus figures. The Santa figures are plump and seem to hover in mid-air as if carried by a breeze.",
        "A figure is hanging from an upside-down forest, holding onto golden leaves. The trees and leaves appear to be suspended above a night sky with a crescent moon below. The inverted composition creates a surreal atmosphere.",
        "A violinist plays in front of a hedge maze, while identically dressed men in dark suits walk through the maze, appearing lost or aimless. The violinist seems to be the only person embracing individuality and creativity in contrast to the uniform crowd.",
        "A man in old-fashioned sailor attire sits on a small wooden raft, drifting through choppy ocean waves. The raft is equipped with a flag and a barrel, and a shark fin can be seen in the water nearby.",
        "A man sits in a small rowboat, fishing in a sea where fish float above the surface, each holding a small umbrella.",
        "A large stack of houses and buildings, precariously balanced in a single towering structure, rises up from the water.",
        "An artist stands on a ladder painting a large frame suspended from a tree, depicting a city skyline. People are seen walking into the frame, transitioning from being headless figures to fully detailed individuals within the painted scene."
    ]
    card = random.choice(cards)
    image_embedding = embedding_extraction(card)
    best_description = get_best_matching_description(image_embedding, descriptions)
    print(f"Best Description: {best_description}")
    clues = GPT_generation(best_description)
    print(clues)
    clues_splitted = clues.split("\n")
    cleaned_clues = [clue.lstrip('0123456789. ').rstrip('.') for clue in clues_splitted if clue]  # Rimuove i numeri e gli spazi
    random_clue = random.choice(cleaned_clues)
    print(f"Random Clue: {random_clue}")
    return random_clue, card
