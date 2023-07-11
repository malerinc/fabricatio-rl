// TODO:
//  1. Finish commenting script!
//  4. Add popup functionality
function render_heatmap(matrix_object, matrix_name, markup, container_id, wip_only=false) {
    let heatdata = matrix_object['data'];
    let y_label = matrix_object['y_label'];
    let x_label = matrix_object['x_label'];
    let n_rows = matrix_object['n_rows'];
    let n_cols = matrix_object['n_cols'];
    let m_type = matrix_object['nfo_type'];
    let domain = [matrix_object['min_value'], matrix_object['max_value']];
    drawHeatmap(markup, heatdata, n_rows, n_cols, domain, container_id,
        matrix_name, x_label, y_label, m_type, wip_only)
}

function split_capitalize(original_string, split_char) {
    let splits = original_string.split(split_char);
    let result = '';
    for (let i=0; i < splits.length; i++) {
        result += splits[i].charAt(0).toUpperCase() + splits[i].slice(1) + ' '
    }
    return result
}

function drawHeatmap(markup, data, n_rows, n_cols, domain, id, matrix_name,
                     xlabel = "Operation Index",
                     ylabel = "Job Index", m_type='job',
                     wip_only=false) {
    let fig_name = split_capitalize(matrix_name, '_');
    // dimension definition
    let margin = {top: 40, right: 10, bottom: 40, left: 50};
    let total_width = 0;
    if (n_cols === 1) {
        total_width = 110;
    } else {
        total_width = 310
    }
    // let total_width = 50 * n_cols + margin.left + margin.right;
    let total_height = 360; // 360
    let svg_width = total_width - margin.left - margin.right;   // 50 px per col + 50px padding for ax labels&co
    let svg_height = total_height - margin.top - margin.bottom;  // 35 px per row + 50px padding for ax labels&co
    // create containers
    let base = create_svg(id, total_width, total_height,
        matrix_name + '_view');
    add_canvas(base, 0, 0, total_width, total_height);
    let svg = base.append('g')
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    let clipid = add_clip_paths(svg, margin, svg_width, svg_height);
    // add axes
    let y_domain = wip_only && (m_type === 'jobs' || m_type === 'tracking') ?
        markup['wip_indices'] : [...data.keys()];

    let axes = add_axes(
        svg.append('g').attr('clip-path', 'url(#clipx' + clipid + ')'),
        svg.append('g').attr('clip-path', 'url(#clipy' + clipid + ')'),
        data, svg_width, svg_height, y_domain);
    // add tiles and text
    let tile_group = svg.append("g").attr('clip-path', 'url(#clip' + clipid + ')');
    let w_tile = axes.scale.x.bandwidth(), h_tile = axes.scale.y.bandwidth();
    let fontsize_num = Math.min(
        (0.95 * w_tile) / (3 * 0.7), (0.8 * h_tile));
    let font_size = '' + fontsize_num + 'px';
    let colorscale = undefined;
    let next_wip_idx = 0;
    for (let i = 0; i < n_rows; i++) {
        if (i === markup['wip_indices'][next_wip_idx]) {
            // get colorscale
            colorscale = build_color_scale(m_type, domain, true);
            next_wip_idx += 1;
        } else {
            colorscale = build_color_scale(m_type, domain, false);
            if (wip_only && (m_type === 'jobs' || m_type === 'tracking')) {
                continue
            }
        }
        for (let j = 0; j < n_cols; j++) {
            let val = n_cols > 1? data[i][j] : data[i];
            let val_string = is_int(val)? val.toString() :
                Number(val).toFixed(2);
            let x_tile = axes.scale.x(j), y_tile = axes.scale.y(i);
            add_colored_tile(tile_group, x_tile, y_tile, w_tile, h_tile,
                colorscale(val), undefined);
            add_text(tile_group, x_tile, y_tile, // x_tile - 5, y_tile + 4.4
                w_tile / 2, h_tile / 2 + fontsize_num / 2.5, val_string, font_size)
        }
    }
    mark_tiles(markup, tile_group, axes, w_tile, h_tile, font_size, m_type);
    // add title and labels
    add_zoom(base, svg_width, svg_height, margin, tile_group,
        axes.ax.x, axes.ax.y, fontsize_num);
    add_title(base, fig_name, total_width / 2, 20);
    add_x_label(base, xlabel, total_width / 2, svg_height + margin.top + 30);
    add_y_label(base, ylabel, - total_height / 2, 0);
}

function add_clip_paths(group_container, margin, svg_width, svg_height){
    let clipid = Math.random().toString();
    group_container.append("clipPath")       // define a clip path
        .attr("id", "clip" + clipid)         // give the clipPath an ID
        .append("rect")                      // shape it as an rectangle
        .attr("x", 0)                        // position the x-centre
        .attr("y", 0)                        // position the y-centre
        .attr("width", svg_width)            // set the x radius
        .attr("height", svg_height);         // set the y radius
    group_container.append('clipPath')
        .attr('id', 'clipx' + clipid)
        .append('rect')
        .attr('x', 0)
        .attr('y', svg_height)
        .attr('width', svg_width)
        .attr('height', margin.bottom);
    group_container.append('clipPath')
        .attr('id', 'clipy' + clipid)
        .append('rect')
        .attr('x', -margin.left)
        .attr('y', -10)
        .attr('width', margin.left+1)
        .attr('height', svg_height+15);
    return clipid
}

function add_zoom(svg, svg_width, svg_height, margin, tile_group, xax, yax,
                  fontsize){
    /**
     * Adds axis aligned zoom functionality to the production state heatmaps. First the
     * zoom parameters are fixed for the passed container. The a handling function is defined
     * to be triggered on mouse wheel events.
     *
     * The handler scales and  moves all tiles&text proportional to the scale and translation
     * attributes of the event parameter respectively. The axes are kept in place along the
     * defined dimension (x axis doesn't move vertically, and the y axis doesn't move
     * horizontally). The axes text is kept at the original size.
     *
     * Note that a clip-path object is needed to keep both axes and tiles contained to a
     * particular view.
     *
     * @see add_clip_paths
     */
    let zoom = d3.zoom()
        .scaleExtent([1, Infinity])
        .translateExtent([[margin.right, 0], [svg_width, svg_height]])
        .extent([[margin.right, 0], [svg_width, svg_height]])
        .on("zoom", transform_elements);
    svg.call(zoom);

    function transform_elements(event) {
        let t = event.transform;
        //xax.scale.range([ 0, svg_width * t.k]);
        tile_group.selectAll(".hm_plot.tile")
            .attr("transform", 'translate(' + t.x + ', ' + t.y + ')scale(' + t.k + ')');
        tile_group.selectAll(".hm_plot.text")
            .attr("transform", 'translate(' + t.x + ', ' + t.y + ')scale(' + t.k + ')');
        // transform x axis
        xax.attr("transform", 'translate(' + t.x + ', ' + svg_height + ')scale(' + t.k + ')');
        xax.selectAll("text")
            .attr("transform", 'scale(' + 1 / t.k + ')')
            .style("font-size", (fontsize * t.k) + "px");
        xax.selectAll("line").attr("transform", 'scale(' + 1/t.k + ')');
        // transform y axis
        yax.attr("transform", 'translate(0, ' + t.y + ')scale(' + t.k + ')');
        yax.selectAll("text")
            .attr("transform", 'scale(' + 1/t.k + ')')
            .style("font-size", (fontsize * t.k) + "px");
        yax.selectAll("line").attr("transform", 'scale(' + 1/t.k + ')');
    }
}

function is_int(num) {
    /**
     * Checks whether a number is an integer by calculating the remainder modulo 1.
     *
     * @param {number} num - The number to be checked.
     * @return {boolean} - True if num is an integer, otherwise false.
     */
    return num % 1 === 0;
}

function create_svg(container_id, width, height, this_id) {
    /**
     * Appends an empty svg object with the width, height and margin as specified by
     * the given parameters to the DOM element identified by the container_id parameter. A margin-top
     * and margin-left of 10px are added as style parameters.
     *
     * Note that object coordinates within an svg start in the upper left corner (0, 0).
     * The positive x-axis extends left of the origin and the positive y-axis downwards.
     *
     * Returns the empty svg object.
     *
     * @param {string} container_id - The DOM element container_id.
     * @param {number} width - The width of the svg.
     * @param {number} height - The height of the svg.
     *
     * @see d3 select/ append / attr / style
     */
    return d3.select("#" + container_id).append("svg").attr("id", this_id)
        .attr("width", width)
        .attr("height", height)
        .style("margin-left", '10px')
        .style("margin-top", '10px')
}

function add_axes(container_x, container_y, data, width, height, y_domain) {
    // Labels of row and columns
    let myGroups = Array.isArray(data[0])?  [...data[0].keys()] : [0];
    // Build x scale and axis:
    let x = d3.scaleBand()
        .range([ 0, width ])
        .domain(myGroups)
        .padding(0.01);
    // Labels of row and columns
    // Build y scale and axis:
    let y = d3.scaleBand()
        .range([ height, 0 ])
        .domain(y_domain)
        .padding(0.01);

    let w_tile = x.bandwidth(), h_tile = y.bandwidth();
    let fontsize_num = Math.min(
        (0.95 * w_tile) / (3 * 0.7), (0.8 * h_tile));

    let xAxis = container_x.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        //.attr('text-anchor', 'end')
        .style("font-size", fontsize_num + "px")
        .call(d3.axisBottom(x));
    let yAxis = container_y.append("g")
        .attr("class", "y axis")
        // .attr("text-anchor", "end")
        .style("font-size", fontsize_num + "px")
        .call(d3.axisLeft(y));
    return {'scale': {'x': x, 'y': y}, 'ax': {'x': xAxis, 'y': yAxis}}
}

function build_color_scale(matrix_type, values_domain, wip=false) {
    let colorscale = undefined;

    if (wip) {
        colorscale = d3.scaleLinear()
            .range(["white", "#06a40a"])
            .domain([values_domain[0],values_domain[1]]);
    } else {
        colorscale = d3.scaleLinear()
        .range(["white", "rgb(133,133,133)"])
        .domain([values_domain[0],values_domain[1]]);
    }
    if (matrix_type === 'tracking') {
        colorscale = d3.scaleLinear()
            .range(["white", "#2369ee"])
            .domain([values_domain[0],values_domain[1]]);
    } else if (matrix_type === 'machines') {
        colorscale = d3.scaleLinear()
            .range(["white", "#ff0202"])
            .domain([values_domain[0],values_domain[1]]);
    }
    return colorscale
}

function mark_tiles(markup, tile_group, axes, w_tile,
                    h_tile, fontsize, matrix_type) {
    if (markup.hasOwnProperty('scheduling_mode')) {
        if (markup.scheduling_mode === 'Sequencing') {
            mark_tiles_sequencing_mode(tile_group, markup, matrix_type,
                axes, w_tile, h_tile, fontsize);
        } else { // mode is routing
            mark_tiles_routing_mode(tile_group, markup, matrix_type,
                axes, w_tile, h_tile, fontsize)
        }
    }
}

function mark_tiles_sequencing_mode(tile_group, markup, matrix_type,
                                    axes, w_tile, h_tile, fontsize) {
    if ((matrix_type==='jobs' || matrix_type === 'tracking')) {
        for (let i = 0; i < markup['legal_actions'].length; i++) {
            let row_idx = markup['legal_actions'][i][0];
            let col_idx = markup['legal_actions'][i][1];
            if (col_idx === markup['wait_scheduling']) {
                continue;
            }
            let frame_color = '#FFFF00';
            if (row_idx === markup['action_taken'][0] &&
                col_idx === markup['action_taken'][1]) {
                frame_color = '#ff0509';
            }
            add_colored_tile(tile_group, axes.scale.x(col_idx),
                axes.scale.y(row_idx), w_tile, h_tile, frame_color,
                fontsize / 10);
        }
    } else { // matrix type is machines
        add_colored_tile(tile_group,
            axes.scale.x(markup['current_machine']),
            axes.scale.y(markup['current_machine']),
            w_tile, h_tile, '#FFFF00',
            fontsize / 10);
    }
}

function mark_tiles_routing_mode(tile_group, markup, matrix_type,
                                 axes, w_tile, h_tile, fontsize) {
    if ((matrix_type==='jobs' || matrix_type === 'tracking')) {
        let alts = markup['current_operation_alternatives'];
        for (let i = 0;
             i < alts.length; i++) {
            let row_idx = alts[i][0], col_idx = alts[i][1];
            let frame_color = '#FFFF00';
            if (row_idx === markup['current_operation'][0] &&
                col_idx === markup['current_operation'][1]) {
                frame_color = '#ff0509';
            }
            add_colored_tile(tile_group, axes.scale.x(col_idx),
                axes.scale.y(row_idx), w_tile, h_tile, frame_color,
                fontsize / 10);
        }
    } else { // matrix type is machines
        let src_machine = markup['current_machine'];
        let destinations = markup['legal_actions'];
        for (let i = 0; i < destinations.length; i++) {
            let dest_machine = parseInt(destinations[i]);
            if (dest_machine === markup['action_taken']) {
                add_colored_tile(tile_group,
                    axes.scale.x(src_machine),
                    axes.scale.y(dest_machine),
                    w_tile, h_tile, '#ff0509',
                    fontsize / 10);
            } else {
                add_colored_tile(tile_group,
                axes.scale.x(src_machine),
                axes.scale.y(dest_machine),
                w_tile, h_tile, '#FFFF00',
                fontsize / 10);
            }
        }
    }
}

function add_colored_tile(svg, x, y, width, height, color,
                          fontsize=undefined) {
    let rectangle = svg.append("rect")  // add the squares
        .attr("class", "hm_plot tile")
        .attr("x", x)
        .attr("y", y)
        .attr("width", width)
        .attr("height", height);

    if (fontsize !== undefined) {
        rectangle.attr('stroke', color)
            .attr('stroke-width', (fontsize / 10) + 'px')
            .attr('fill', color).attr('opacity', 0.5)
    } else {
        rectangle.style("fill", color);
    }
}

function add_x_label(svg, xlabel, x, y) {
    svg.append("text")
        .attr("class", "x label")
        .attr("text-anchor", "middle")
        .attr("x", x)
        .attr("y", y)
        .style("font-weight", 700)
        .style("font-size", "14px")
        .style("text-anchor", "top")
        .text(xlabel);
}

function add_y_label(svg, ylabel, x, y) {
    /**
     * Adds the y-axis label. The text is rotated -90°.
     */
    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", y)
        .attr("x", x)
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .style("font-weight", 700)
        .style("font-size", "14px")
        .text(ylabel);
}
