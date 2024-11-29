package search_engine.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/")
@CrossOrigin(origins = "http://localhost:8081/")
public class SearchController {
    @PostMapping("/search")
    public String FromTextSearchImage(@RequestBody String text){
        System.out.println(text);
        return "ok";
    }
}
