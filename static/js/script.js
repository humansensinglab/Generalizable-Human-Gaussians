// scripts.js
let slideIndices = {
    slider1: 0,
    slider2: 0
};

function changeSlide(n, sliderId) {
    showSlides(slideIndices[sliderId] += n, sliderId);
}

function showSlides(n, sliderId) {
    let slider = document.getElementById(sliderId);
    let slides = slider.getElementsByClassName("video-slide");

    if (n >= slides.length) {
        slideIndices[sliderId] = 0; // If reached end, go to first slide
    }

    if (n < 0) {
        slideIndices[sliderId] = slides.length - 1; // If reached start, go to last slide
    }

    // Hide all slides
    for (let i = 0; i < slides.length; i++) {
        slides[i].classList.remove("active");
    }

    // Show the current slide
    slides[slideIndices[sliderId]].classList.add("active");
}

// Initialize both sliders
showSlides(0, 'slider1');
showSlides(0, 'slider2');

let autoplay = true;
let autoplayInterval = 6500; // Change slides every 5 seconds

if (autoplay) {
    setInterval(() => {
        changeSlide(1, 'slider1');
        changeSlide(1, 'slider2');
    }, autoplayInterval);
}