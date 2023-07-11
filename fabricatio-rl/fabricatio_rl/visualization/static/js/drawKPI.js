function drawKPI(id, svg_id, y_max, data){
    // set the dimensions and margins of the graph
    var margin = {top: 50, right: 100, bottom: 60, left: 50};
    var width_total = 500;
    width = width_total - margin.left - margin.right;
    height = 380 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    var svg = d3.select("#" + id)
    .append("svg").attr("id", svg_id)
    .attr("width", width_total)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");


    // Box
    svg.append('rect')
      .attr('x', -40)
      .attr('y', -20)
      .attr('width', width_total)
      .attr('height', 340)
      .attr('stroke', 'black')
      .attr('fill', 'white')

	// Figure name
    svg.append("text")
        .attr("x", (width / 2) + 20)
        .attr("y", 300)
        .attr("text-anchor", "middle")
        .style("font-size", "14px")
        .style("font-weight", 700)
        .style("text-decoration", "underline")
        .text("KPI");


      // List of subgroups
      let keys = []
      for (let k in data[0]) keys.push(k);
      keys.pop();
      let subgroups = keys;

      // List of groups
      var groups = data.map(a => a.index);
      // Add X axis
      var x = d3.scaleBand()
          .domain(groups)
          .range([0, width])
          .padding([0.2])
      svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x).tickSize(0));

      // Add Y axis
      var y = d3.scaleLinear()
        .domain([0, y_max])
        .range([ height, 0 ]);
      svg.append("g")
        .call(d3.axisLeft(y));

      // Another scale for subgroup position?
      var xSubgroup = d3.scaleBand()
        .domain(subgroups)
        .range([0, x.bandwidth()])
        .padding([0.05])

      // color palette = one color per subgroup
      var color = d3.scaleOrdinal()
        .domain(subgroups)
        .range([
            "#3957ff", "#c9080a", "#0b7b3e", "#0bf0e9", "#fd9b39",
            "#888593", "#906407", "#98ba7f", "#fe6794", "#10b0ff",
            "#964c63", "#1da49c", "#0ad811", "#bbd9fd", "#fe6cfe",
            "#297192", "#d1a09c", "#78579e", "#81ffad", "#739400", "#ca6949"])

      // Show the bars
      svg.append("g")
        .selectAll("g")
    // Enter in data = loop group per group
    .data(data)
    .enter()
    .append("g")
      .attr("transform", function(d) {
          return "translate(" + x(d.index) + ",0)";
      })
    .selectAll("rect")
    .data(function(d) { return subgroups.map(function(key) { return {key: key, value: d[key]}; }); })
    .enter().append("rect")
      .attr("x", function(d) { return xSubgroup(d.key); })
      .attr("y", function(d) { return y(d.value); })
      .attr("width", xSubgroup.bandwidth())
      .attr("height", function(d) { return height - y(d.value); })
      .attr("fill", function(d) { return color(d.key); });

    // Add one dot in the legend for each name.
    var size = 10
    svg.selectAll("mydots")
    .data(subgroups)
    .enter()
    .append("rect")
    .attr("x", width_total - 130)
    .attr("y", function(d,i){ return i*(size+5)})
    .attr("width", size)
    .attr("height", size)
    .style("fill", function(d){ return color(d); })

    // Add one dot in the legend for each name.
    svg.selectAll("mylabels")
    .data(subgroups)
    .enter()
    .append("text")
    .attr("x", width_total - 130 + size * 1.3)
    .attr("y", function(d,i){ return i*(size+5) + (size/2)})
    .style("fill", function(d){ return color(d); })
    .text(function(d){ if(data.length == 2) return "Machine: "+d;
                       else return "Job: "+d; })
    .attr("text-anchor", "left")
    .attr("font-size", "12px")
    .attr("font-weight", 700)
    .style("alignment-baseline", "middle")

}