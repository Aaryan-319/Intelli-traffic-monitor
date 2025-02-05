document.addEventListener("DOMContentLoaded", function () {
    const faqItems = document.querySelectorAll(".faq-item");
    
    faqItems.forEach(item => {
        item.querySelector(".faq-question").addEventListener("click", function () {
            item.classList.toggle("active");
        });
    });
  
    document.querySelectorAll(".slide-in").forEach(element => {
        element.style.display = "none";
        setTimeout(() => element.style.display = "block", 0);
        element.classList.add("slide-down");
    });
  
    document.querySelectorAll(".fade-in").forEach(element => {
        element.style.display = "none";
        setTimeout(() => element.style.display = "block", 0);
        element.classList.add("fade-in-effect");
    });
  });
  
  document.getElementById("contactForm").addEventListener("submit", function(event) {
    event.preventDefault();
    alert("Thank you for your message! We'll get back to you soon.");
  });
  
  
  document.addEventListener("DOMContentLoaded", function() {
    const navLinks = document.querySelectorAll('.nav-links li a');
  
    navLinks.forEach(link => {
        if (link.href === window.location.href) {
            link.classList.add('active');
        }
    });
  });