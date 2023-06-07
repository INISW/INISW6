
function fold_card() {
    $('.topic-cards').animate({
        height: '60px',
        width: '750px',
        marginTop: '10px',
        marginBottom: '0px'
    });
    
    $('.card-body').animate({
        top: '60%'
    })

    $('#expand-text').animate({
        height: '0px',
        opacity: '0'
    })

    $('.card-body').children('hr').animate({
        opacity: '0'
    })
    $('.card-overlay').animate({
        opacity: '0.6'
    })

    $('.gap').animate({
        height: '10vh'
    })

    $('.card-title').animate({
        fontSize: '18px'
    })

    $('#gap-expand').animate({
        height: '10vh'
    })

    $('#expand-body').css('display', 'grid')

    $('#expand-body').css('flex', '1')
}

