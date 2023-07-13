
const divElement = document.getElementsByClassName("header-navigator")[0]; // 获取目标div元素
divElement.style.display = "block"; // 将display属性设置为block,以显示元素

let navigator_links = document.querySelectorAll('.header-navigator ul li a[href^="#"]');
navigator_links.forEach(link => {
    link.addEventListener('click', function (event) {
        event.preventDefault();
        let target = document.querySelector(this.getAttribute('href'));
        target.scrollIntoView({ behavior: 'smooth', block: 'start', inline: 'nearest' });
        // 修改网页 URL
        let url = window.location.href.split('#')[0];
        let newUrl = url + this.getAttribute('href');
        history.pushState(null, null, newUrl);

    });
});

function isScrolledIntoView(elem) {
    var docViewTop = window.pageYOffset;
    var docViewBottom = docViewTop + window.innerHeight;
    var elemTop = elem.offsetTop;
    var elemBottom = elemTop + elem.offsetHeight;
    return ((elemTop <= docViewBottom) && (elemBottom >= docViewTop));
}

var headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
var previousHeading;

window.addEventListener('scroll', function () {
    var found = false;
    for (var heading of headings) {
        if (!found && isScrolledIntoView(heading)) {
            var heading_id = heading.id;
            var link = document.querySelector(`a[href="#${heading_id}"]`);
            if (link) {
                link.style.fontWeight = 'bold';
                previousHeading = heading;
                found = true;
            }
        } else {
            var heading_id = heading.id;
            var link = document.querySelector(`a[href="#${heading_id}"]`);
            if (link) {
                link.style.fontWeight = 'normal';
            }
        }
    }

    // If no heading is found, set the previous heading to bold
    if (!found && previousHeading) {
        var previousLink = document.querySelector(`a[href="#${previousHeading.id}"]`);
        if (previousLink) {
            previousLink.style.fontWeight = 'bold';
        }
    }
});