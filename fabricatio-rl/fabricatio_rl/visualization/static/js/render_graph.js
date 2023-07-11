function render_graph(id, data, title, svg_id) {
    let width = 650;
    let height = 360;

    // Create a new directed graph
    let g = new dagreD3.graphlib.Graph().setGraph({});
    // Add nodes
    for(let i = 0; i < data.nodes.length; i++){
        g.setNode("" + data.nodes[i].id, {
            label: data.nodes[i].name, shape: "circle"
        })
    }

    // add edges
    for(let i = 0; i < data.links.length; i++){
        g.setEdge("" + data.links[i].source, "" + data.links[i].target, {});
    }

    // Create the renderer
    let render = new dagreD3.render();

    // Outer Container
    let svg = d3.select("#"+id)
      .append("svg").attr("id", svg_id)
      .attr("width", width)
      .attr("height", height);

    // The white frame :)
    svg.append('rect')
      .attr('x', 10)
      .attr('y', 0)
      .attr('width', width - 10)
      .attr('height', height)
      .attr('stroke', 'black')
      .attr('fill', 'white');

    // the inner container
    let svgGroup = svg.append("g");

    // Set up zoom support
    let zoom = d3.zoom().on("zoom", function(event) {
          svgGroup.attr('transform', event.transform);
        });
    svg.call(zoom);

    // Run the renderer. This is what draws the final graph.
    render(svgGroup, g);

    // Calculate scale and translation parameters
    let gw = g.graph().width;
    let gh = g.graph().height;
    let initialScale = Math.min(
        svg.attr('width') / (gw + 2 * 30),
        svg.attr('height') / (gh + 2 * 20));
    // x, y such that graph is centered in containing svg
    setup_zoom(svgGroup, svg, gw, gh, initialScale);

    // move labels inside the nodes
    svg.selectAll("circle").attr("x", 0).attr("y", 0).attr("r", 12);
    svg.selectAll("text").attr("x", 0).attr("y", 0);
    svg.selectAll("tspan").attr("dy", 2.5).attr("x", -10);

     // Figure name
    svg.append("text")
      .attr("x", (width / 2 + 20))
      .attr("y", height - 40)
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .style("font-weight", 700)
      .style("text-decoration", "underline")
      .text(title);
}