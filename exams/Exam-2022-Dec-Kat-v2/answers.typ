
#let images = (
    "images/dtu_logo.png",

)

#let load_images() = {
    for value in images {
        align(center)[#image(value)]
    }
}

#load_images()