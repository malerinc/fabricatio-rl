function drawSlider(mx, def) {
    let container = $("#slider");
    container.html("");
    let slider = d3.sliderHorizontal()
        .min(0)
        .max(mx)
        .step(1)
        .width(container.innerWidth())
        .default(def)
        .tickFormat(d3.format("d"))
        .ticks(tickFrequency(mx)).displayValue(false)
        .on('onchange', (val) => {
            $("#vis_root").children().empty();
           update_page(val);
           $('#current_id').text(val);
           d3.select('p#value').text("State: "+ d3.format('')(val));
           $("#gr"+val).show();
        });

    d3.select('#slider')
        .append('svg')
        .attr("preserveAspectRatio", "xMinYMin meet")
        .attr("viewBox", "0 0 " + container.innerWidth() + " 100")
        .append('g')
        .attr('width', '100%')
        .attr('height', 100)
        .attr('transform', 'translate(20,30)scale(0.95)')
        .call(slider);

    d3.select("#slider").attr("align","center");
}

function tickFrequency(sz){
    let tf = 10;
    if(sz < 10){
        tf =  sz;
    }
    return tf;
}
