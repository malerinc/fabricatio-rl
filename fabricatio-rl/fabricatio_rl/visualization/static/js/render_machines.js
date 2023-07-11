function render_machines(id, data) {
    let width = 350; let height = 360;
    // // mouse over
    // let tooltip = d3.select("#"+id)
    //   .append("div")
    //   .style("position", "fixed")
    //   .style("z-index", "10")
    //   .style("visibility", "hidden")
    //   .style("background-color", "#e5eedc")
    //   .style("border", "solid")
    //   .style("border-width", "2px")
    //   .style("border-radius", "5px")
    //   .style("padding", "5px");
    let svg = d3.select("#"+id)
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("id", "machines");
    add_canvas(svg, 10, 0, width - 10, height);
    add_title(svg, "Machine State", width / 2 + 20, height - 40);
    let svgGroup = svg.append("g");
    let dims = get_layout_parameters(data.nodes.length);
    let idx_col = 0; let idx_row = 0;  //current col/row idx
    // TODO: offsets dynamically?
    let offset_x = 250, offset_y = 140;
    // TODO: dynamic n_row, n_col computation
    let path_positions = render_nodes(svgGroup, data.nodes, idx_col, idx_row,
        offset_x, offset_y, dims);
    render_links(svgGroup, data.links, path_positions);
    // Set up zoom support
    let zoom = d3.zoom().on("zoom", function(event) {
          svgGroup.attr('transform', event.transform);
        });
    svg.call(zoom);
    // Calculate scale and translation parameters
    let gw = dims.n_cols * offset_x;
    let gh = dims.n_rows * offset_y;
    let initialScale = Math.min(
        svg.attr('width') / (gw + 2 * 30),
        svg.attr('height') / (gh + 2 * 20));
    // x, y such that graph is centered in containing svg
    // a wee bit more on the left ^^
    setup_zoom(svgGroup, svg, gw, gh, initialScale);
}

function construct_machine_table(svgGroup, node, posX, posY) {
    let colors = [
        "#ffffff", "#3957ff", "#c9080a", "#0b7b3e", "#0bf0e9",
        "#fd9b39", "#888593", "#906407", "#98ba7f", "#fe6794",
        "#10b0ff", "#964c63", "#1da49c", "#0ad811", "#bbd9fd",
        "#fe6cfe", "#297192", "#d1a09c", "#78579e", "#81ffad",
        "#739400", "#ca6949"];

    let op_b1 = '\xa0', op_b2 = '\xa0', op_b3 = '\xa0';  // nbsp ;)
    let op_a = node['operation_queue'].length === 0 ? '\xa0' :
        node['operation_processed'][0];
    let n_ops = node['operation_queue'].length;
    if (n_ops > 3) {
        op_b1 = node['operation_queue'][0];
        op_b2 = node['operation_queue'][1];
        op_b3 = ' ... ';
    } else if (n_ops === 3) {
        op_b1 = node['operation_queue'][0];
        op_b2 = node['operation_queue'][1];
        op_b3 = node['operation_queue'][2];
    } else if (n_ops === 2) {
        op_b1 = node['operation_queue'][0];
        op_b2 = node['operation_queue'][1];
    } else if (n_ops === 1) {
        op_b1 = node['operation_queue'][0];
    }
    let table = svgGroup.append("foreignObject")
            .attr("x", posX).attr("y", posY)
            .attr("width", "250 ")
            .attr("height", "90")
            .append("xhtml:body");
    table = table.append("table").attr("class", "machine_state");
    // first row
    let row1 = table.append('tr');
    row1.append("td")
        .attr("class", "m_name").text('M' + node.id)
        .attr("style", "background-color: "+colors[node.id]);
    row1.append("td").attr("colspan", "3")
        .text('Capabilities: ' + node['machine_group']);
    // second row
    let row2 = table.append('tr');
    row2.append("td").attr("colspan", "3")
        .text('Input Buffer (' + node['operation_queue'].length + ')');
    row2.append("td").text('Processing:');
    // third row
    let row3 = table.append('tr');
    row3.append("td").text(op_b1);
    row3.append("td").text(op_b2);
    row3.append("td").text(op_b3);
    row3.append("td").attr("class", "btm").text(op_a);
}

function get_layout_parameters(nr_items){
    let n_cols, n_rows;
    if(nr_items <= 4) {
        n_rows = 2; n_cols = 2;
    }
    else {
        n_rows = Math.round(nr_items / 3) + 1;
        n_cols = 3;
    }
    return {'n_rows': n_rows, 'n_cols': n_cols}
}

function render_nodes(container, nodes, idx_col, idx_row,
                      offset_x, offset_y, dims) {
    let path_positions = {};
    for(let i = 0; i < nodes.length; i++){
        let node =  nodes[i];
        let posX = idx_col * offset_x;
        let posY = idx_row * offset_y;
        path_positions['' +  nodes[i].id] = {
            'x': posX + 100,
            'y': posY
        };
        construct_machine_table(container, node, posX, posY);
        idx_col = (idx_col + 1) % dims.n_cols;
        if (idx_col === 0) {
            idx_row = (idx_row + 1) % dims.n_rows
        }
    }
    return path_positions
}

function render_links(container, links, path_positions) {
    for(let i = 0; i < links.length; i++){
        let src = undefined;
        if(links[i].source === -1) {
            src = 0;
        }
        else {
            src = links[i].source;
        }
        let tgt = links[i].target;
        let source_pos = path_positions[src];
        let target_pos = path_positions[tgt];
        drawPath(container, src, tgt, source_pos, target_pos,
            links[i]['route_chosen'], links[i]['op_routed']);
    }
}

function get_turning_point_position(src, tgt, src_pos) {
    let src_rn = Math.floor(src / 3); let src_cn = src % 3;
    let tgt_rn = Math.floor(tgt / 3); let tgt_cn = tgt % 3;
    let midp_posX = undefined;
     if(src === tgt){
        midp_posX = src_pos.x + 125
    } else if(src_cn === tgt_cn){
        if(tgt_rn - src_rn === 1){
            midp_posX = src_pos.x
        } else {
            midp_posX = src_pos.x + 125
        }
    } else if(src_cn > tgt_cn){
        midp_posX = src_pos.x - 125
    } else {
        midp_posX = src_pos.x + 125;
    }
     return midp_posX
}

function get_line_attributes(route_chosen) {
    let color, dashstroke;
    if (route_chosen === null) { // we are sequencing
        color = "#960509"; dashstroke = false;
    } else if (route_chosen) {  //corresponds to current action
        color = "#00ff13"; dashstroke = false;
    } else {
        color = "#00ab13"; dashstroke = true;
    }
    return {'color': color, 'dashstroke': dashstroke}
}

function add_non_overlapping_text() {
    // TODO!!!
}

function drawPath(svgGroup, src, tgt, src_pos, tgt_pos, route_chosen, op_routed) {
    // TODO: dynamic nr cols!!!
    let turning_point_line_x = get_turning_point_position(src, tgt, src_pos);
    let attrs = get_line_attributes(route_chosen);

    svgGroup.append('rect').attr("class", "hm_plot path")
        .attr("x", src_pos.x - 2.5)
        .attr("y", src_pos.y + 80)
        .attr("width", 5)
        .attr("height", 10)
        .attr('stroke', 'black')
        .attr('fill', "#960509");
    // TODO> loop! ;)
    add_line(svgGroup,
        src_pos.x, src_pos.y + 90,
        src_pos.x, src_pos.y + 120,
        attrs.color, attrs.dashstroke);
    add_line(svgGroup,
        src_pos.x, src_pos.y + 120,
        turning_point_line_x, src_pos.y + 120,
        attrs.color, attrs.dashstroke);
    add_line(svgGroup,
        turning_point_line_x, src_pos.y + 120,
        turning_point_line_x, tgt_pos.y - 20,
        attrs.color, attrs.dashstroke);
    add_line(svgGroup,
        turning_point_line_x, tgt_pos.y - 20,
        tgt_pos.x, tgt_pos.y - 20,
        attrs.color, attrs.dashstroke);
    add_text(svgGroup, turning_point_line_x, tgt_pos.y - 50,
        (tgt_pos.x - turning_point_line_x) / 2, 20,
        '' + op_routed, 14 );
    add_line(svgGroup,
        tgt_pos.x, tgt_pos.y - 20,
        tgt_pos.x, tgt_pos.y + 5,
        attrs.color, attrs.dashstroke);
    svgGroup.append('circle')
        .attr('cx', tgt_pos.x)
        .attr('cy', tgt_pos.y)
        .attr('r', 5)
        .attr('stroke', 'black')
        .attr('fill', attrs.color);
}
