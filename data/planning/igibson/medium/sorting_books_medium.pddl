(define (problem sorting_books_0)
    (:domain igibson)

    (:objects
     	hardback_1 - movable
    	table_1 - object
    	shelf_1 - object
    	book_1 - movable
    )
    
    (:init 
        (ontop hardback_1 table_1) 
    )
    
    (:goal 
        (and 
            (ontop hardback_1 shelf_1)
            (ontop book_1 shelf_1)
        )
        
    )
)