from PIL import Image, ImageDraw, ImageFont

def stitch_images_horizontally(name, output_path, title):
    filenames = [
        name+"/frame_0000.png",
        name+"/frame_0015.png",
        name+"/frame_0035.png",
        name+"/frame_0050.png"
    ]

    images = [Image.open(f) for f in filenames]
    
    # Get the dimensions of the images
    widths, heights = zip(*(img.size for img in images))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    # Font settings
    font_size = 32
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    title_height = font.getsize(title)[1] + 10  # Adding some padding for the title
    
    # Create a new image with the appropriate size
    stitched_image = Image.new('RGB', (total_width, max_height + title_height), (255, 255, 255))
    
    # Draw the title on the new image
    draw = ImageDraw.Draw(stitched_image)
    text_width, text_height = draw.textsize(title, font=font)
    draw.text(((total_width - text_width) / 2, 5), title, fill="black", font=font)
    
    x_offset = 0
    for img in images:
        stitched_image.paste(img, (x_offset, title_height))
        x_offset += img.width
    
    # Save the stitched image
    stitched_image.save(output_path)

# Stitch the specified images and save the result with a title
stitch_images_horizontally("animation_frames_R2", "Images/Small_bu.png", "a) Small Upper Bound")
stitch_images_horizontally("animation_frames_R8", "Images/Large_bu.png", "b) Large Upper Bound")
stitch_images_horizontally("animation_frames_R2_8", "Images/Increasing_bu.png", "c) Increasing Upper Bound")
