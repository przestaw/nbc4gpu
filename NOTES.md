### Problemy wstępne

- boost.compute jest nierozwijany od 2 lat
- brak szczegółowej dokumentacji w przystępnej formie
- większość przykładów używa C98 i OpenCL
- najczęstrze użycia :
    - wrapper do wołania surowego kodu OpenCL
    - wraper do OpenCl i używanie funkcji do transferu wektoróœ danych/typów graficznych [vec3/vec4]

### Problemy z konfiguracją

- brak instrukcji co do kompilacji pod windows 
- brak jasnych informacji o sterownikach OpenCL od producentów kart graficznych [Nvidia intel]

### Problemy na etapie implementacji 

- Błędy kompilacji dynamiczne [riuntime] w przypadku użytkowania liczb ujemnych ? [brak szczegółowych informacji o powodach wystąpienia, brak info w dokumentacji]
- Dostępność w zasadzie jedynie pobierznej dokumentacji zaimplementowanych makr [kompilowanych do openCl jak transform, reduce, accumulate]
    - https://www.boost.org/doc/libs/1_65_1/libs/compute/doc/html/boost_compute/reference.html
    - przykłady użycia najczęściej są pustymi ramkami, brak informacji o wymogach początkowych
    - funkcje wbudowane jak "+" "-" "*" "exp" nie są przedstawione w dokumentacji
    
    
### prezentacja

https://raw.githubusercontent.com/boostcon/cppnow_presentations_2015/master/files/Boost.ComputeCxxNow2015.pdf