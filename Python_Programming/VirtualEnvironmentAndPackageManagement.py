
# Virtual Enviroment(Sanal Ortam)
'''
 İzole çalışma ortamları oluşturmak için kullanılan araçlardır.
 Farklı çalışmalar için oluşabilecek kütüphane ve versiyonlar için çalışmalar
 birbirini etkilemeyecek şekilde oluşturma imkanı sunar.
 conda, pipenv,venv,virtualenv
'''

# Package Management (Paket Yönetimi) Araçları
'''
 pip
 pipenv
 conda (hem sanal ortam hem de paket yöneticisi)
'''

# Sanal Ortamlar ve Paket Yönetimi Arasındaki İlişki
'''
 venv ve virtualenv paket yönetim aracı olarak pip kullanıyor.
 conda ve pipenv hem paket hem de sanal ortam yönetiminde kullanılıyor.
 pip paket yönetimi için kullanılır.
'''
# Sanal ortamların listelenmesi
# conda env list

# Sanal ortam oluşturma
# conda create -n myenv

# Sanal ortamı aktif etme
# conda activate myenv

# Yüklü paketlerin listelenmesi:
# conda list

# Paket yükleme:
# conda install numpy

# Aynı anda birden fazla paket yükleme:
# conda install numpy scipy pandas

# Paket silme:
# conda remove package_name

# Belirli bir versiyona göre paket yükleme:
# conda install numpy=1.20.1

# Paket yükseltme:
# conda upgrade numpy

# Tüm paketlerin yükseltilmesi:
# conda upgrade –all

# pip: pypi (python package index) paket yönetim aracı

# Paket yükleme:
# pip install pandas

# Paket yükleme versiyona göre:
# pip install pandas==1.2.1