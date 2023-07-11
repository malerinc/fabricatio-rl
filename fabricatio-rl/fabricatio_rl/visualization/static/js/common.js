function add_line(group, start_x, start_y, end_x, end_y, color,
                  dash_stroke=false) {
    let line = group.append("line").attr("class", "hm_plot path")
        .attr("x1", start_x)
        .attr("y1", start_y)
        .attr("x2", end_x)
        .attr("y2", end_y)
        .attr("stroke-width", 3).attr("stroke", color);
    if (dash_stroke) {
        line.attr("stroke-dasharray", 3)
    }
}

function setup_zoom(svgGroup, svg, gw, gh, initialScale) {
    let inner_margin_x = (svg.attr('width') - gw * initialScale) / 2 + 20;
    let inner_margin_y = (svg.attr('height') - gh * initialScale) / 2 + 5;
    // append scale and translation attributes
    let translate_string = 'translate(' +
        inner_margin_x / 2 + ',' + inner_margin_y + ')';
    let scale_string = 'scale(' + initialScale + ')';
    svgGroup.attr('transform', translate_string + scale_string);
}

function add_title(svg, title, x, y) {
    svg.append("text")
        .attr("x", x)
        .attr("y", y)
        .attr("text-anchor", "middle")
        .style("font-weight", 700)
        .style("font-size", "14px")
        .style("text-decoration", "underline")
        .text(title);
}

function add_canvas(svg, x, y, width, height) {
    /**
     * Adds a white background box with a black frame to the passed svg element giving the
     * appearence of a canvas. Elements subsequently added to the svg will be drawn ontop
     * of this canvas.
     *
     * @param {Object} svg - The svg object to add the canvas to.
     * @param {number} x - The x coordinate of the upper left corner of the canvas relative
     *      to the containing svg upper left corner.
     * @param {number} y - The y coordinate of the upper left corner of the canvas relative
     *      to the containing svg upper left corner.
     * @param {number} width - The width of the canvas box.
     * @param {number} height - The height of the canvas box.
     *
     * @see create_svg
     */
    svg.append('rect')
        .attr('x', x)
        .attr('y', y)
        .attr('width', width)
        .attr('height', height)
        .attr('stroke', 'black')
        .attr('fill', '#ffffff');
}

function add_text(svg, x, y, anchor_dx, anchor_dy, text, font_size) {
    svg.append("text")
        .attr("class", 'hm_plot text')
        .attr("x", x)
        .attr("y", y)
        .attr("dx", anchor_dx)
        .attr("dy", anchor_dy)
        .style("font-size", font_size)  // 70% of the tile height
        .attr("font-weight", 700)
        .attr("text-anchor", "middle")
        .text(text);
}